"""Generate standard ArUco bitmap assets with white margin, at a specified print DPI."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

p=argparse.ArgumentParser()
p.add_argument('--size-mm',type=float,default=100)
p.add_argument('--dpi',type=int,default=300)
p.add_argument('--output',default='outputs/markers')
a=p.parse_args()
if not np.isfinite(a.size_mm) or not 30<=a.size_mm<=300 or not 72<=a.dpi<=600:
    p.error('标志边长须为30—300 mm，DPI须为72—600。')
directory=Path(a.output);directory.mkdir(parents=True,exist_ok=True)
side=round(a.size_mm*a.dpi/25.4);margin=round(side*.25)
dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for i in range(4):
    marker=cv2.aruco.generateImageMarker(dictionary,i,side)
    paper=np.full((side+2*margin,side+2*margin),255,np.uint8)
    paper[margin:margin+side,margin:margin+side]=marker
    Image.fromarray(paper).save(directory/f'aruco_4x4_50_id{i}_{a.size_mm:g}mm.png',dpi=(a.dpi,a.dpi))
print('Generated 4 markers. Print at 100% scale; measure the black outer square, excluding white margin.')
