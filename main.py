import argparse
import json
from pathlib import Path
from vision.pipeline import inspect
from vision.quality_check import Options


def main():
    p=argparse.ArgumentParser(description="PC构件平面尺寸测量PoC，需要实测的四标志布局。")
    p.add_argument("image")
    p.add_argument("--layout",required=True)
    p.add_argument("--component-id",default="PC-W01")
    p.add_argument("--design-length-mm",type=float,default=4000)
    p.add_argument("--design-width-mm",type=float,default=1000)
    p.add_argument("--marker-size-mm",type=float,default=100)
    p.add_argument("--tolerance-mm",type=float,default=50)
    p.add_argument("--manual-corners",help="JSON文件，四组EXIF归一化原图像素坐标，沿边界顺序")
    p.add_argument("--selected-marker-ids",nargs=4,type=int)
    p.add_argument("--output",default="outputs/latest")
    a=p.parse_args()
    try:
        layout=json.loads(Path(a.layout).read_text(encoding="utf-8-sig"))
        corners=json.loads(Path(a.manual_corners).read_text(encoding="utf-8-sig")) if a.manual_corners else None
    except (OSError,json.JSONDecodeError) as exc:
        p.error(f"无法读取布局或角点JSON：{exc}")
    options=Options(a.design_length_mm,a.design_width_mm,a.marker_size_mm,a.tolerance_mm,a.component_id)
    result=inspect(a.image,layout,options,a.output,corners,a.selected_marker_ids)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if result["success"] else 2


if __name__=="__main__":
    raise SystemExit(main())
