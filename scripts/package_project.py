"""Build a shareable source bundle from an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
GUIDE = '''PC 构件尺寸测量 — 收件人使用说明

这是可运行源码演示包，不是免安装 EXE。

一、运行（Windows 10/11，64 位）
1. 将整个 ZIP 解压到普通本地文件夹，不要直接在压缩包内运行。
2. 安装 Python 3.12（64 位），安装时勾选 Add Python to PATH。
   官方下载入口：https://www.python.org/downloads/windows/
3. 双击 launch.cmd。首次启动会联网下载依赖，请保持窗口开启。
4. 浏览器自动打开 http://127.0.0.1:8765 。若未打开，手动访问此地址。
5. 点击“运行模拟演示”，即可查看自动测量、边界标注及透视校正。
6. 服务窗口保持开启；按 Ctrl+C 停止服务。

二、可以演示什么
- 模拟照片自动测量长宽，显示与设计值的偏差。
- 手动修正四角，拖动或输入坐标后重新测量。
- 下载本次 JSON 结果。
- 上传真实照片时，需要另行准备实测布局 JSON 并确认共面条件。

三、关键说明
- 无需 aTrust 或元筑账号即可运行本地演示。
- 模拟样例的标志黑边为 150 mm，四中心布局为 4400 × 1400 mm。
  这些是模拟数据，不能直接套用任意现场摆放。
- 真实照片测量需打印尺寸正确、已知标志相对位置，且标志与被测表面齐平。
- PDF 打印选择 A4、实际大小/100%，打印后核对黑色方块横竖均为150 mm。
- 当前已完成本地原型和接口，尚未完成真实现场精度验证或元筑平台接入。
- 仅本机访问，不能将本版本直接部署成企业公网服务。

四、包内内容
app/、api/、vision/：工作台、接口、测量算法。
outputs/simulation/marker_150mm/input.png：可直接运行的模拟照片。
output/pdf/：四页标准标志打印版。
outputs/markers/：四张标志 PNG。
docs/：拍摄规范、验证记录、现场记录表、接口说明。
tests/：测量与接口测试。
MANIFEST.json：文件清单与 SHA-256 校验值。

五、常见问题
- 提示找不到 Python：安装 Python 3.12 并重试。
- 下载依赖失败：检查网络后重新双击 launch.cmd；不要忽略安装错误。
- 8765 端口冲突：关闭占用该端口的其他程序后重试。
- 真实照片显示重拍/人工处理：按页面提示核对标志和边界，不代表服务启动失败。

本包未包含企业原始资料、培训附件、aTrust 安装程序、账号密码、真实现场照片、
当前机器的虚拟环境、历史检测输出或 Git 历史。
'''

def build():
    paths = []
    for folder, patterns in {'app':['*.html','*.css','*.js'], 'api':['*.py'], 'vision':['*.py'], 'tests':['*.py'], 'config':['layout.example.json']}.items():
        for pattern in patterns:
            paths.extend((ROOT/folder).glob(pattern))
    for name in ['README.md','requirements.txt','pytest.ini','main.py','launcher.py','launch.cmd','start-workbench.ps1',
                 'samples/components.csv','scripts/generate_markers.py','scripts/run_simulation.py',
                 'docs/03_PoC方案与拍摄规范.md','docs/04_验证记录.md','docs/05_现场测量记录表.md','docs/06_本地工作台与接口.md',
                 'outputs/simulation/marker_150mm/input.png','output/pdf/ArUco_150mm_ID0-3_打印版.pdf']:
        paths.append(ROOT/name)
    paths.extend((ROOT/'outputs/markers').glob('aruco_4x4_50_id[0-3]_150mm.png'))
    contents = {p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(set(paths))}
    contents['先读我.txt'] = GUIDE.encode('utf-8-sig')
    # Included README must only link to documentation in this shareable bundle.
    readme=contents['README.md'].decode('utf-8')
    readme='\n'.join(line for line in readme.splitlines() if '(docs/01_' not in line and '(docs/02_' not in line)
    readme=readme.replace('当前电脑已创建 `.venv` 并安装依赖；无需再次安装。','收件人请先按“先读我.txt”双击 launch.cmd 完成首次安装。')
    contents['README.md']=readme.encode('utf-8')
    manifest=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in sorted(contents.items())]
    contents['MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    directory=ROOT/'deliverables';directory.mkdir(exist_ok=True)
    destination=directory/f'PC构件尺寸测量_演示源码包_{datetime.now():%Y%m%d_%H%M%S}.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(contents.items()):
            z.writestr('PC-measurement-demo/'+name,data)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(contents)
    print(json.dumps({'zip':str(destination),'files':len(contents),'bytes':destination.stat().st_size},ensure_ascii=False))
    return destination

if __name__=='__main__':
    build()
