"""Experimental metric rectification using coplanar equal-size square cards.

Relative card positions and orientations are NOT known or fitted to a component
dimension. Fit five metric homography degrees of freedom from square side and
diagonal distances. Output must be checked on unseen photos before production.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from .quality_check import InspectionError
from .component_detector import ordered_quad
from .perspective import transform

PAIRS = [(0,1,1.), (1,2,1.), (2,3,1.), (3,0,1.), (0,2,2**.5), (1,3,2**.5)]


def fit_free_squares(quads, marker_size_mm=150):
    q = np.asarray(quads, np.float64)
    if q.ndim != 3 or q.shape[1:] != (4,2) or len(q)<3 or not np.isfinite(q).all():
        raise InspectionError('INSUFFICIENT_CUSTOM_MARKERS', '试验标定至少需要三个可靠且分散的方形标志。')
    if not np.isfinite(marker_size_mm) or marker_size_mm<=0:
        raise InspectionError('INVALID_MARKER_SIZE', '标志边长无效。', 'INVALID_INPUT')
    for quad in q:
        ordered_quad(quad)
    center = q.reshape(-1,2).mean(axis=0)
    scale = max(np.ptp(q.reshape(-1,2),axis=0).max(), 1)
    n = (q-center)/scale
    centers = n.mean(axis=1).astype(np.float32)
    if abs(cv2.contourArea(cv2.convexHull(centers))) < .01:
        raise InspectionError('POOR_MARKER_DISTRIBUTION', '标志中心过于集中或近似共线。')
    def matrix(x):
        return np.array([[np.exp(x[0]),x[1],0.],[0.,np.exp(x[2]),0.],[x[3],x[4],1.]])
    def residual(x):
        h=matrix(x)
        hp=np.concatenate([n,np.ones((*n.shape[:2],1))],axis=2) @ h.T
        den=hp[:,:,2]
        mm=hp[:,:,:2]/np.maximum(den[:,:,None],.02)
        errors=np.asarray([(np.linalg.norm(mm[:,a]-mm[:,b],axis=1)-marker_size_mm*factor) for a,b,factor in PAIRS]).T
        # Penalize a projective horizon crossing the marker region.
        return np.r_[errors.ravel(), np.minimum(den-.15,0).ravel()*10000]
    edge=np.linalg.norm(n-np.roll(n,-1,axis=1),axis=2).mean()
    initial=np.array([np.log(marker_size_mm/edge),0.,np.log(marker_size_mm/edge),0.,0.])
    fit=least_squares(residual,initial,loss='soft_l1',f_scale=1.,max_nfev=1200,x_scale='jac')
    if not fit.success or not np.isfinite(fit.x).all():
        raise InspectionError('CUSTOM_FIT_FAILED','方形标志联合标定没有收敛。')
    norm=np.array([[1/scale,0,-center[0]/scale],[0,1/scale,-center[1]/scale],[0,0,1.]])
    h=matrix(fit.x)@norm
    normalized=transform(q,h)
    errors=np.asarray([np.linalg.norm(normalized[:,a]-normalized[:,b],axis=1)-marker_size_mm*f for a,b,f in PAIRS]).T
    metrics={'marker_count':len(q),'square_distance_rmse_mm':float(np.sqrt(np.mean(errors**2))),
             'max_square_distance_error_mm':float(np.max(abs(errors))),
             'per_marker_max_error_mm':np.max(abs(errors),axis=1).tolist(),
             'known_marker_size_mm':float(marker_size_mm),'uses_known_marker_layout':False}
    return h, metrics


def measure_with_free_squares(quads, boundary, marker_size_mm=150):
    boundary=ordered_quad(boundary)
    h,metrics=fit_free_squares(quads,marker_size_mm)
    denominators=np.c_[boundary,np.ones(4)]@h[2]
    if denominators.min()<=.15:
        raise InspectionError('PROJECTIVE_HORIZON','标定在构件范围内不稳定。')
    mm=transform(boundary,h)
    if not cv2.isContourConvex(mm.astype(np.float32)):
        raise InspectionError('INVALID_RECTIFIED_BOUNDARY','校正后的边界无效。')
    vectors=np.roll(mm,-1,axis=0)-mm
    edges=np.linalg.norm(vectors,axis=1)
    length,width=sorted([(edges[0]+edges[2])/2,(edges[1]+edges[3])/2],reverse=True)
    metrics['opposite_edge_difference_mm']=float(max(abs(edges[0]-edges[2]),abs(edges[1]-edges[3])))
    metrics['max_angle_cosine']=float(np.max(abs(np.sum(vectors*np.roll(vectors,-1,axis=0),axis=1)/(edges*np.roll(edges,-1)))))
    # Leave one marker out: monitor sensitivity without any tape-measure input.
    variations=[]
    if len(quads)>=4:
        for index in range(len(quads)):
            try:
                other=np.delete(np.asarray(quads),index,axis=0)
                lh,_=fit_free_squares(other,marker_size_mm)
                lm=transform(boundary,lh);le=np.linalg.norm(lm-np.roll(lm,-1,axis=0),axis=1)
                ll,lw=sorted([(le[0]+le[2])/2,(le[1]+le[3])/2],reverse=True)
                variations.append(float(max(abs(ll-length),abs(lw-width))))
            except InspectionError:
                variations.append(None)
    metrics['leave_one_out_dimension_change_mm']=variations
    reasons=[]
    if metrics['max_square_distance_error_mm']>7.5:reasons.append('MARKER_GEOMETRY_INCONSISTENT')
    if metrics['max_angle_cosine']>.12 or metrics['opposite_edge_difference_mm']>50:reasons.append('BOUNDARY_GEOMETRY_INCONSISTENT')
    if not variations or any(v is None or v>50 for v in variations):reasons.append('CALIBRATION_SENSITIVE_OR_UNCHECKED')
    return {'status':'REVIEW' if reasons else 'EXPERIMENTAL_MEASUREMENT',
            'measured_length_mm':float(length),'measured_width_mm':float(width),
            'quality_reasons':reasons,'calibration':metrics,'image_to_mm':h.tolist(),
            'calibration_corners_px':np.asarray(quads,float).tolist(),
            'boundary_mm':mm.tolist(),'boundary_source':'manual_pilot_annotation',
            'limitation':'开发集上的试验结果；不是训练好的模型输出，也不代表现场验收通过。'}


def measure_with_guarded_marker_update(original_quads, refined_quads, boundary, marker_size_mm=150):
    """Reject a corner refinement that destabilizes the existing calibration.

    This compares two predictions only; reference/tape dimensions are not inputs.
    Lower square residual alone cannot justify a large change in component size.
    """
    baseline=measure_with_free_squares(original_quads,boundary,marker_size_mm)
    try:
        candidate=measure_with_free_squares(refined_quads,boundary,marker_size_mm)
        change=max(abs(candidate[k]-baseline[k]) for k in ('measured_length_mm','measured_width_mm'))
        worse=candidate['calibration']['max_square_distance_error_mm']>max(2,baseline['calibration']['max_square_distance_error_mm']*1.1)
        reasons=[]
        if change>50:reasons.append('SIZE_CHANGE_EXCEEDS_50MM')
        if worse:reasons.append('MARKER_RESIDUAL_WORSENED')
        guard={'accepted':not reasons,'max_dimension_change_mm':float(change),'reasons':reasons}
    except InspectionError as exc:
        candidate=None
        guard={'accepted':False,'reasons':[exc.code]}
    result=candidate if guard['accepted'] else baseline
    result['marker_update_guard']=guard
    if not guard['accepted']:
        result['status']='REVIEW'
        result['quality_reasons'].append('MARKER_REFINEMENT_REJECTED')
    return result


def measure_with_selective_marker_update(original_quads, refined_quads, boundary, marker_size_mm=150):
    """Experimental recovery of useful individual updates after joint rejection.

    Selection uses geometry only, never tape dimensions. All partial updates
    remain REVIEW; consistency improvement is not proof of physical accuracy.
    """
    from itertools import combinations
    result = measure_with_guarded_marker_update(original_quads, refined_quads, boundary, marker_size_mm)
    if result['marker_update_guard']['accepted']:
        return result
    original = np.asarray(original_quads, float)
    refined = np.asarray(refined_quads, float)
    changed = [i for i in range(len(original)) if np.max(abs(original[i]-refined[i])) > 1e-6]
    if len(changed) < 2 or len(changed) > 6:
        return result
    base = result['calibration']
    def sensitivity(metrics):
        values = metrics['leave_one_out_dimension_change_mm']
        return max(values) if values and all(v is not None for v in values) else float('inf')
    best = None
    for count in range(1, len(changed)):
        for indices in combinations(changed, count):
            candidate_quads = original.copy()
            candidate_quads[list(indices)] = refined[list(indices)]
            try:
                candidate = measure_with_free_squares(candidate_quads, boundary, marker_size_mm)
            except InspectionError:
                continue
            metrics = candidate['calibration']
            change = max(abs(candidate[k]-result[k]) for k in ('measured_length_mm', 'measured_width_mm'))
            if (change > 50 or metrics['max_square_distance_error_mm'] > base['max_square_distance_error_mm']
                    or metrics['square_distance_rmse_mm'] >= base['square_distance_rmse_mm']-1e-6
                    or sensitivity(metrics) > sensitivity(base)
                    or metrics['opposite_edge_difference_mm'] > max(50, base['opposite_edge_difference_mm'])
                    or metrics['max_angle_cosine'] > max(.12, base['max_angle_cosine'])):
                continue
            if best is None or metrics['square_distance_rmse_mm'] < best['calibration']['square_distance_rmse_mm']:
                candidate['selective_marker_update'] = {'accepted_indices': list(indices), 'max_dimension_change_mm': float(change)}
                best = candidate
    if best is None:
        return result
    best['marker_update_guard'] = result['marker_update_guard']
    best['status'] = 'REVIEW'
    best['quality_reasons'].append('PARTIAL_MARKER_UPDATE_REQUIRES_REVIEW')
    return best
