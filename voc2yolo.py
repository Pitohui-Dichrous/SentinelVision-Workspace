import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# ------- 配置区 -------
# 确认当前脚本所在目录为根目录
ROOT = Path(__file__).parent.resolve()

# VOC原始目录结构
ANNOTATIONS = ROOT / 'data' / 'helmetdata' / 'Annotations'
IMAGES      = ROOT / 'data' / 'helmetdata' / 'JPEGImages'

# YOLO输出目录
OUT_IMAGES = ROOT / 'data' / 'helmetdata' / 'images'
OUT_LABELS = ROOT / 'data' / 'helmetdata' / 'labels'
# -----------------------

# 检查输出目录存在
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

# 类名映射，按名称列表顺序填写
# 目前根据github，['person', 'hat', 'fire'] → nc=3
NAMES = ['person', 'hat', 'fire']

def convert_box(size, box):
    """把VOC (xmin,ymin,xmax,ymax) 转为YOLO (x_center,y_center,w,h)"""
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    w_norm = (xmax - xmin) / w
    h_norm = (ymax - ymin) / h
    return x_center, y_center, w_norm, h_norm

def xml_to_yolo(xml_path, txt_path):
    """解析单个VOC xml，列出YOLO txt"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in NAMES:
            continue  # 跳过不在关心类别中的对象
        cls_id = NAMES.index(cls)

        bndbox = obj.find('bndbox')
        box = (
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text),
        )
        x, y, w_norm, h_norm = convert_box((w, h), box)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w_norm:.6f} {h_norm:.6f}")

    # 即使无任何目标，也写一个空文件，让所有图片都参与训练
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main():
    xml_files = sorted(ANNOTATIONS.glob('*.xml'))
    print(f"Found {len(xml_files)} XML files, start converting…")

    for xml in xml_files:
        stem = xml.stem  # 文件名不含后缀，比如 'part2_000123'
        img_candidates = list(IMAGES.glob(f"{stem}.*"))  # 支持 .jpg/.jpeg/.png
        if not img_candidates:
            print(f"⚠️ WARNING: 无法找到对应图片：{stem}")
            continue
        img_path = img_candidates[0]
        # 复制图片到YOLO images/
        dst_img = OUT_IMAGES / img_path.name
        shutil.copyfile(img_path, dst_img)

        # 生成labels/
        txt_path = OUT_LABELS / f"{stem}.txt"
        xml_to_yolo(xml, txt_path)

    print("✅ 转换完成，所有图片和标签都已写入：")
    print(f"   images/  ← {OUT_IMAGES}")
    print(f"   labels/  ← {OUT_LABELS}")

if __name__ == '__main__':
    main()
# 头大三圈，祖宗别再出错了
#
#                        _oo0oo_
#                       o8888888o
#                       88" . "88
#                       (| -_- |)
#                       0\  =  /0
#                     ___/`---'\___
#                   .' \\|     |// '.
#                  / \\|||  :  |||// \
#                 / _||||| -:- |||||- \
#                |   | \\\  - /// |   |
#                | \_|  ''\---/''  |_/ |
#                \  .-\__  '-'  ___/-. /
#              ___'. .'  /--.--\  `. .'___
#           ."" '<  `.___\_<|>_/___.' >' "".
#          | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#          \  \ `_.   \_ __\ /__ _/   .-` /  /
#      =====`-.____`.___ \_____/___.-`___.-'=====
#                        `=---='
#
#
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#            佛祖保佑       永不宕机     永无BUG
