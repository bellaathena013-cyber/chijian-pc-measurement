"""Refine existing numbered-card candidates using the outer dark-frame edge.

Only the outside-light to inside-dark transition is accepted. The white inner
frame has the reverse polarity and must never become the 150 mm scale edge.
"""
import cv2
import numpy as np
from .custom_markers import order_quad, pattern_score
from .segmentation_refinement import _intersection, _line


def refine_marker_candidate(gray, candidate):
    original=order_quad(candidate['corners_px']).astype(float)
    lengths=np.linalg.norm(original-np.roll(original,-1,axis=0),axis=1)
    radius=max(2,min(8,float(lengths.min())*.1))
    offsets=np.arange(-radius,radius+.01,.5)
    lines=[];evidence=[]
    source=gray.astype(np.float32)
    for a,b in zip(original,np.roll(original,-1,axis=0)):
        direction=(b-a)/np.linalg.norm(b-a)
        inward=np.array([-direction[1],direction[0]])
        points=a+(b-a)*np.linspace(.15,.85,64)[:,None]
        coords=points[:,None,:]+offsets[None,:,None]*inward
        def sample(delta):
            p=(coords+delta*inward).astype(np.float32)
            return cv2.remap(source,p[:,:,0],p[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        # Correct polarity is essential: do not fit to the white inner frame.
        gradient=sample(-.75)-sample(.75)
        profile=np.median(gradient,axis=0)
        best=int(np.argmax(profile));score=float(profile[best])
        info={'contrast':score,'offset_px':float(offsets[best])}
        others=abs(offsets-offsets[best])>=3
        ambiguous=others.any() and profile[others].max()>score*.85
        if score<12 or best in (0,len(offsets)-1) or ambiguous:
            return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'weak_or_ambiguous_edge','edges':evidence+[info]}}
        local=np.where((abs(offsets-offsets[best])<=1)[None,:],gradient,-1e6)
        picks=np.argmax(local,axis=1)
        strong=gradient[np.arange(len(points)),picks]>max(12,score*.5)
        if strong.sum()<32:
            return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'insufficient_edge_support'}}
        line=_line(coords[np.arange(len(points)),picks][strong])
        if abs(line[:2]@inward)<.98:
            return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'edge_angle_changed'}}
        lines.append(line);evidence.append(info)
    intersections=[_intersection(lines[i-1],lines[i]) for i in range(4)]
    if any(p is None for p in intersections):
        return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'unstable_corners'}}
    refined=np.array(intersections,np.float32)
    shift=float(np.linalg.norm(refined-original,axis=1).max())
    area_ratio=cv2.contourArea(refined)/cv2.contourArea(original.astype(np.float32))
    if not cv2.isContourConvex(refined) or shift>max(3,lengths.min()*.12) or not .75<area_ratio<1.25:
        return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'excessive_geometry_change'}}
    score,parts,_=pattern_score(gray,refined)
    if score<.68:
        return {**candidate,'outer_edge_refinement':{'status':'kept_original','reason':'pattern_check_failed'}}
    return {**candidate,'corners_px':refined.tolist(),'pattern_score':score,'pattern_parts':parts,
            'outer_edge_refinement':{'status':'refined','max_corner_shift_px':shift,'edges':evidence}}


def refine_marker_candidates(image, candidates):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    return [refine_marker_candidate(gray,c) for c in candidates]


def calibrated_marker_candidates(candidates, measurement):
    """Expose the exact corners actually used, including guard fallbacks."""
    corners = measurement['calibration_corners_px']
    if len(corners) != len(candidates):
        raise ValueError('Calibration and marker candidate counts differ')
    result = []
    for candidate, used in zip(candidates, corners):
        points = np.asarray(used, float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('Invalid calibration corners')
        result.append({**candidate, 'proposed_corners_px': candidate['corners_px'],
                       'corners_px': points.tolist(), 'corners_source': 'actual_calibration',
                       'min_edge_px': float(np.linalg.norm(points-np.roll(points,-1,axis=0),axis=1).min())})
    return result
