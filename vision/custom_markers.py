"""Experimental detector for the supplied black/white numbered square cards.

This is pattern verification, not OCR or ArUco decoding. Candidate indices are
NOT the printed numbers. No component dimensions participate in detection.
"""
import cv2
import numpy as np


def order_quad(points):
    p = np.asarray(points, np.float32).reshape(4, 2)
    order = np.argsort(np.arctan2(p[:, 1]-p[:, 1].mean(), p[:, 0]-p[:, 0].mean()))
    p = p[order]
    return np.roll(p, -np.argmin(p.sum(axis=1)), axis=0)


def pattern_score(gray, quad):
    dest = np.float32([[0, 0], [95, 0], [95, 95], [0, 95]])
    patch = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad.astype(np.float32), dest), (96, 96))
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yy, xx = np.mgrid[:96, :96] / 95.
    r = np.maximum(abs(xx-.5), abs(yy-.5))
    diagonal = abs(abs(xx-.5)-abs(yy-.5)) < .035
    border = (r > .46) & (r < .49)
    band = (r > .36) & (r < .41) & ~diagonal
    inner = (r > .23) & (r < .30)
    scores = [float((binary[border] == 0).mean()), float((binary[band] > 0).mean()), float((binary[inner] == 0).mean())]
    return min(scores), scores, patch


def detect_custom_markers(image, max_dimension=1800, contour_tolerance=.025):
    if not np.isfinite(max_dimension) or max_dimension < 96:
        raise ValueError('max_dimension must be finite and at least 96')
    if not np.isfinite(contour_tolerance) or not 0 < contour_tolerance <= .05:
        raise ValueError('contour_tolerance must be in (0, .05]')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1., max_dimension/max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    masks = [cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]]
    masks.extend(cv2.threshold(small, level, 255, cv2.THRESH_BINARY_INV)[1] for level in (70, 120, 160, 190))
    masks.append(cv2.morphologyEx(cv2.Canny(small, 40, 100), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    masks.extend(cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, size, 7) for size in (31, 61))
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if not 70 < area < small.size*.035:
                continue
            p = cv2.approxPolyDP(contour, contour_tolerance*cv2.arcLength(contour, True), True)
            if len(p) != 4 or not cv2.isContourConvex(p):
                continue
            p = order_quad(p) / scale
            lengths = np.linalg.norm(p-np.roll(p, -1, axis=0), axis=1)
            if lengths.min() < 15 or lengths.max()/lengths.min() > 6:
                continue
            score, parts, _ = pattern_score(gray, p)
            proposal_source = 'outer_contour'
            if score < .68 and parts[0] < .4 and parts[1] > .4 and parts[2] > .7:
                # The supplied artwork has a narrow outer black frame. Use its
                # inner white-frame contour only as a search proposal, then
                # refine the *outer* corner on the original image. Never use
                # the white boundary directly as the 150 mm measurement edge.
                canonical = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
                h = cv2.getPerspectiveTransform(canonical, p.astype(np.float32))
                outer = cv2.perspectiveTransform(((canonical-.5)*(15/13)+.5).reshape(1, 4, 2), h)[0]
                if (outer < 5).any() or (outer[:,0] > gray.shape[1]-6).any() or (outer[:,1] > gray.shape[0]-6).any():
                    continue
                outer = cv2.cornerSubPix(gray, outer.reshape(-1, 1, 2).copy(), (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, .01)).reshape(4, 2)
                oscore, oparts, _ = pattern_score(gray, outer)
                if oscore > score:
                    p, score, parts = outer, oscore, oparts
                    proposal_source = 'white_frame_proposal_refined'
            if score < .68:
                continue
            # Refine the outer black corners at source resolution.
            refined = cv2.cornerSubPix(gray, p.reshape(-1, 1, 2).copy(), (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, .01)).reshape(4, 2)
            refined_score, refined_parts, _ = pattern_score(gray, refined)
            if refined_score >= score:
                p, score, parts = refined, refined_score, refined_parts
            center = p.mean(axis=0)
            match = next((j for j, q in enumerate(candidates) if np.linalg.norm(center-np.mean(q['corners_px'], axis=0)) < lengths.min()*.4), None)
            item = {'corners_px': p.tolist(), 'pattern_score': score, 'pattern_parts': parts, 'proposal_source': proposal_source, 'min_edge_px': float(np.linalg.norm(p-np.roll(p, -1, axis=0), axis=1).min())}
            if match is None:
                candidates.append(item)
            elif score > candidates[match]['pattern_score']:
                candidates[match] = item
    candidates.sort(key=lambda r: np.mean(r['corners_px'], axis=0)[1])
    for index, item in enumerate(candidates):
        item['candidate_index'] = index
    return candidates


def detect_custom_markers_robust(image):
    """Recover slightly irregular contours without replacing baseline corners.

    Both passes use the same black/white pattern checks. The second pass is
    only a proposal search; its score is not a calibrated probability.
    """
    candidates = detect_custom_markers(image)
    for item in candidates:
        item['detection_pass'] = 'baseline'
    # Extra, less regular contours can destabilize an existing calibration.
    # Recovery is reserved for insufficient detections, never an automatic
    # replacement of an already computable measurement.
    if len(candidates) >= 3:
        return candidates
    for item in detect_custom_markers(image, contour_tolerance=.04):
        center = np.mean(item['corners_px'], axis=0)
        if any(np.linalg.norm(center-np.mean(other['corners_px'], axis=0)) <
               .5*max(item['min_edge_px'], other['min_edge_px']) for other in candidates):
            continue
        item['detection_pass'] = 'irregular_contour_recovery'
        candidates.append(item)
    candidates.sort(key=lambda item: np.mean(item['corners_px'], axis=0)[1])
    for index, item in enumerate(candidates):
        item['candidate_index'] = index
    return candidates
