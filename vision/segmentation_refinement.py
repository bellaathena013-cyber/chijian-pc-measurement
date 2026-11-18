"""Conservative mask cleanup and image-supported planar boundary fitting.

Inputs are pixels and a predicted mask only. No tape dimensions or labels are
accepted. Irregular shapes and ambiguous edges remain reviewable failures.
"""
import cv2
import numpy as np
from .custom_markers import order_quad


def _intersection(a, b):
    matrix=np.stack([a[:2],b[:2]])
    if abs(np.linalg.det(matrix))<.05:
        return None
    return np.linalg.solve(matrix,-np.array([a[2],b[2]]))


def _line(points):
    vx,vy,x,y=cv2.fitLine(np.asarray(points,np.float32),cv2.DIST_WELSCH,0,.01,.01).ravel()
    return np.array([-vy,vx,vy*x-vx*y],float)


def _edge_from_pixels(gray, a, b, radius):
    direction=(b-a)/np.linalg.norm(b-a)
    normal=np.array([-direction[1],direction[0]])
    samples=a+(b-a)*np.linspace(.12,.88,80)[:,None]
    offsets=np.arange(-radius,radius+1,dtype=float)
    coords=samples[:,None,:]+offsets[None,:,None]*normal
    def sample(delta):
        p=(coords+normal*delta).astype(np.float32)
        return cv2.remap(gray,p[:,:,0],p[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    gradient=np.abs(sample(1.5)-sample(-1.5))
    profile=np.median(gradient,axis=0)
    best=int(np.argmax(profile))
    score=float(profile[best])
    separated=np.abs(offsets-offsets[best])>=5
    runner=float(profile[separated].max()) if separated.any() else 0.
    info={'contrast':score,'offset_px':float(offsets[best]),'alternative_contrast':runner}
    if score<8:
        return None,{**info,'status':'weak'}
    if runner>score*.85:
        return None,{**info,'status':'ambiguous'}
    if best in (0,len(offsets)-1):
        return None,{**info,'status':'search_limit'}
    allowed=np.abs(offsets-offsets[best])<=2
    local=np.where(allowed[None,:],gradient,-1)
    picks=np.argmax(local,axis=1)
    strong=gradient[np.arange(len(samples)),picks]>=max(8,score*.5)
    if strong.sum()<40:
        return None,{**info,'status':'weak'}
    points=coords[np.arange(len(samples)),picks][strong]
    fitted=_line(points)
    if abs(np.dot(fitted[:2],normal))<.995:
        return None,{**info,'status':'unstable_direction'}
    return fitted,{**info,'status':'refined','support_samples':int(strong.sum())}


def refine_component_mask(mask, image=None, use_image_edges=True):
    raw=np.asarray(mask)
    if raw.ndim!=2 or not np.isfinite(raw).all():
        raise ValueError('Mask must be a finite two-dimensional array')
    binary=(raw>0).astype(np.uint8)
    h,w=binary.shape
    if image is not None and image.shape[:2]!=(h,w):
        raise ValueError('Image and mask dimensions must match')
    result={'status':'MANUAL_REQUIRED','reasons':[],'quad_px':None,'mask':binary,
            'metrics':{'input_foreground_px':int(binary.sum())}}
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:
        result['reasons']=['EMPTY_MASK'];return result
    areas=stats[1:,cv2.CC_STAT_AREA]
    largest=int(np.argmax(areas))+1
    cleaned=(labels==largest).astype(np.uint8)
    removed=1-float(cleaned.sum()/binary.sum())
    result['mask']=cleaned
    result['metrics'].update(component_count=count-1,removed_foreground_fraction=removed)
    if cleaned.sum()<max(100,h*w*.01):
        result['reasons']=['TARGET_TOO_SMALL'];return result
    if removed>.05:
        result['reasons']=['MULTIPLE_SIGNIFICANT_REGIONS'];return result
    contours,_=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea)
    points=contour.reshape(-1,2).astype(float)
    if (points.min(axis=0)<2).any() or (points.max(axis=0)>[w-3,h-3]).any():
        result['reasons']=['TARGET_CLIPPED'];return result
    hull=cv2.convexHull(contour)
    hull_area=cv2.contourArea(hull)
    solidity=cv2.contourArea(contour)/max(hull_area,1)
    result['metrics']['solidity']=float(solidity)
    if solidity<.94:
        result['reasons']=['NON_RECTANGULAR_OR_CONTAMINATED'];return result
    quad=None
    # A rounded predicted corner makes an inscribed approximation too small.
    # This is only an initial direction proposal, not the accepted boundary;
    # robust side intersections must still pass the final mask-overlap gate.
    for ratio in (.005,.01,.015,.02,.03,.04):
        approx=cv2.approxPolyDP(hull,ratio*cv2.arcLength(hull,True),True)
        if len(approx)==4 and cv2.isContourConvex(approx) and abs(cv2.contourArea(approx)-hull_area)/hull_area<.18:
            quad=order_quad(approx.reshape(4,2)).astype(float);break
    if quad is None:
        result['reasons']=['NO_STABLE_QUADRILATERAL'];return result
    lengths=np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
    band=max(3,min(15,float(lengths.min())*.035))
    lines=[]
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        direction=(b-a)/np.linalg.norm(b-a)
        normal=np.array([-direction[1],direction[0]])
        along=(points-a)@direction/np.linalg.norm(b-a)
        near=(np.abs((points-a)@normal)<band)&(along>.12)&(along<.88)
        if near.sum()<12:
            result['reasons']=['INSUFFICIENT_STRAIGHT_EDGE'];return result
        lines.append(_line(points[near]))
    fitted=[_intersection(lines[i-1],lines[i]) for i in range(4)]
    if any(p is None for p in fitted):
        result['reasons']=['ILL_CONDITIONED_CORNERS'];return result
    fitted=np.array(fitted)
    gray=None
    if image is not None and use_image_edges:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        gray=cv2.GaussianBlur(gray.astype(np.float32),(3,3),.8)
        pixel_lines=[];evidence=[]
        for i,(a,b) in enumerate(zip(fitted,np.roll(fitted,-1,axis=0))):
            refined,info=_edge_from_pixels(gray,a,b,max(5,min(14,round(lengths.min()*.04))))
            evidence.append(info)
            pixel_lines.append(refined if refined is not None else lines[i])
        result['metrics']['image_edges']=evidence
        if any(e['status'] in ('ambiguous','search_limit') for e in evidence):
            result['reasons'].append('AMBIGUOUS_IMAGE_EDGE')
        if any(e['status'] in ('weak','unstable_direction') for e in evidence):
            result['reasons'].append('WEAK_IMAGE_EDGE')
        refined=[_intersection(pixel_lines[i-1],pixel_lines[i]) for i in range(4)]
        if all(p is not None for p in refined):fitted=np.array(refined)
    displacement=float(np.max(np.linalg.norm(fitted-quad,axis=1)))
    result['metrics']['max_corner_adjustment_px']=displacement
    valid=np.isfinite(fitted).all() and cv2.isContourConvex(fitted.astype(np.float32))
    if not valid or displacement>max(12,float(lengths.min())*.1):
        result['reasons'].append('EXCESSIVE_BOUNDARY_CHANGE');return result
    if (fitted<2).any() or (fitted>[w-3,h-3]).any():
        result['reasons'].append('TARGET_CLIPPED');return result
    candidate=np.zeros_like(cleaned)
    cv2.fillPoly(candidate,[np.round(fitted).astype(np.int32)],1)
    overlap=float(np.logical_and(candidate,cleaned).sum()/np.logical_or(candidate,cleaned).sum())
    result['metrics']['quad_mask_iou']=overlap
    if overlap<.94:
        result['reasons'].append('QUAD_DISAGREES_WITH_MASK');return result
    result['quad_px']=fitted.tolist()
    result['status']='REVIEW' if result['reasons'] else 'EXPERIMENTAL_BOUNDARY'
    return result
