from os import path
from typing import NamedTuple
import pathlib

PATH = '/media/max/Transcend/max/plate_recognition/licence_plate_recognition/' \
       'data/raw/plate_detection_external_datasets/data/ocr_yolo/data/train/'

class YoloAnnotationXYWH(NamedTuple):
    object_class: int
    x_center: int
    y_center: int
    width: int
    height: int

class YoloAnnotationXYWHNorm(NamedTuple):
    object_class: int
    x_center: float
    y_center: float
    width: float
    height: float

def xywh2xywhn2(ann_xywh: YoloAnnotationXYWH, w, h) -> YoloAnnotationXYWHNorm:
    return(YoloAnnotationXYWHNorm(
        object_class = ann_xywh.object_class,
        x_center = round((ann_xywh.x_center + 0.5 * ann_xywh.width) / w, 6),
        y_center = round((ann_xywh.y_center + 0.5 * ann_xywh.height) / h, 6),
        width = round(ann_xywh.width / w, 6),
        height = round(ann_xywh.height / h, 6)
        ))

def num(s: str) -> float:
    try:
        return float(int(s))
    except ValueError:
        return float(s)

def read_ann_file(text_file: path) -> str:
    with open(text_file, 'r', encoding='utf8') as f:
        data = f.readlines()
    return data

def get_annotation(raw_annotation: str) -> YoloAnnotationXYWH:
    filtred_raw = raw_annotation[:-1].split(' ')
    return YoloAnnotationXYWH(
            object_class = int(filtred_raw[0]),
            x_center = int(filtred_raw[1]),
            y_center = int(filtred_raw[2]),
            width = int(filtred_raw[3]),
            height = int(filtred_raw[4])
            )

def parse_ann_file(raw_annotation: str) -> list[YoloAnnotationXYWH]:
    full_image_annotation = []
    for ann in raw_annotation:
        full_image_annotation.append(get_annotation(ann))
    return full_image_annotation

def write_anat_file(text_file, normalized_list):
    with open(text_file, 'w', encoding='utf8') as f:
        for anat in normalized_list:
            str_anat = ' '.join([str(i) for i in anat])
            f.write(str_anat)
            f.write('\n')            

def normalize_annotation(data_folder, w=1920, h=1080):
    for text_file in pathlib.Path(path.join(data_folder, 'labels/')).glob('*.txt'):
        raw_annotation = read_ann_file(text_file)
        print(raw_annotation)
        annotation_list = parse_ann_file(raw_annotation)
        print(annotation_list)
        normalized_list = []
        for anat in annotation_list:
            normalized_anat = xywh2xywhn2(anat, w, h)
            print(normalized_anat)
            normalized_list.append(normalized_anat)
        write_anat_file(path.join(data_folder, 'norm_labels', text_file.name), normalized_list)
        print(normalized_list)


if __name__ == '__main__':
    normalize_annotation(PATH, 1920, 1080)

