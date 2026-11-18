"""Independent synthetic camera fixtures; truth is never passed to auto measurement."""
import cv2
import numpy as np
from PIL import Image
from vision.visualization import save_image


def scene(path, length=4000, width=1000, marker_size=100, resolution=(2800,1400),
          perspective=.12, rotate=0, shadow=False, blur=0, missing=(), contrast=145,
          shifted_marker=None, exif=False, component=True):
    scale=.8
    img=np.full((1440,3840,3),60,np.uint8)
    # World frame covers [-200,4600] x [-200,1600] mm; markers surround the component.
    def px(x,y): return np.array([(x+200)*scale,(y+200)*scale])
    corners=np.array([px(2200-length/2,700-width/2),px(2200+length/2,700-width/2),
                      px(2200+length/2,700+width/2),px(2200-length/2,700+width/2)])
    if component:
        cv2.fillConvexPoly(img,corners.astype(np.int32),(60+contrast,)*3)
    if shadow:
        shade=np.linspace(.68,1,img.shape[1])[None,:,None]
        img=np.clip(img*shade,0,255).astype(np.uint8)
        if shadow=='partial':
            img[500:950,1000:2200]=(img[500:950,1000:2200]*.72).astype(np.uint8)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    centres=[(0,0),(4400,0),(4400,1400),(0,1400)]
    for marker_id,(x,y) in enumerate(centres):
        if marker_id in missing: continue
        if shifted_marker==marker_id: x+=80
        c=px(x,y); side=int(round(marker_size*scale))
        left,top=(c-side/2).astype(int); margin=int(side*.25)
        cv2.rectangle(img,(left-margin,top-margin),(left+side+margin,top+side+margin),(255,255,255),-1)
        tag=cv2.aruco.generateImageMarker(dictionary,marker_id,side)
        img[top:top+side,left:left+side]=cv2.cvtColor(tag,cv2.COLOR_GRAY2BGR)
    w,h=resolution
    dst=np.float32([[w*(.06+perspective),h*.10],[w*.94,h*(.1+perspective*.3)],
                    [w*(.94-perspective*.2),h*.90],[w*.06,h*(.90-perspective*.3)]])
    source=np.float32([[0,0],[3839,0],[3839,1439],[0,1439]])
    matrix=cv2.getPerspectiveTransform(source,dst)
    camera=cv2.warpPerspective(img,matrix,(w,h),borderValue=(40,40,40))
    truth=cv2.perspectiveTransform(corners.astype(np.float64)[None],matrix)[0]
    if rotate and rotate%90==0:
        turns=rotate//90
        for _ in range(turns):
            old_h=camera.shape[0]
            camera=cv2.rotate(camera,cv2.ROTATE_90_CLOCKWISE)
            truth=np.column_stack((old_h-1-truth[:,1],truth[:,0]))
    elif rotate:
        old_h,old_w=camera.shape[:2]
        rotation=cv2.getRotationMatrix2D((old_w/2,old_h/2),rotate,1)
        cos,sin=abs(rotation[0,0]),abs(rotation[0,1])
        new_w,new_h=int(old_h*sin+old_w*cos),int(old_h*cos+old_w*sin)
        rotation[0,2]+=new_w/2-old_w/2;rotation[1,2]+=new_h/2-old_h/2
        camera=cv2.warpAffine(camera,rotation,(new_w,new_h),borderValue=(40,40,40))
        truth=np.column_stack((truth,np.ones(4)))@rotation.T
    if blur:
        camera=cv2.GaussianBlur(camera,(0,0),blur)
    if exif:
        # Stored sideways; EXIF 6 restores the camera image and the truth coordinates.
        rgb=cv2.cvtColor(cv2.rotate(camera,cv2.ROTATE_90_COUNTERCLOCKWISE),cv2.COLOR_BGR2RGB)
        im=Image.fromarray(rgb);e=im.getexif();e[274]=6
        im.save(path,exif=e,quality=98)
    else:
        save_image(path,camera)
    return truth
