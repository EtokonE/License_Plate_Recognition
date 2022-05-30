"""
Формат аннотация датасета OCR:
    frame000060.txt
    --------------
    vehicles: 2
position_vehicle: 768 122 555 478
	type: car
plate: ARG4068
position_plate: 1115 551 93 34
	char 1: 1123 564 9 17
	char 2: 1133 564 11 18
	char 3: 1144 564 10 16
	char 4: 1160 562 9 16
	char 5: 1170 562 10 16
	char 6: 1181 562 10 16
	char 7: 1191 561 10 16

position_vehicle: 1403 0 348 190
	type: car
plate: AZI4586
position_plate: 1582 127 65 25
	char 1: 1587 133 7 12
	char 2: 1594 134 9 12
	char 3: 1603 134 5 13
	char 4: 1613 136 7 12
	    char 5: 1620 136 9 13
	char 6: 1628 137 8 12
	char 7: 1636 138 7 11


Нужно преобразовать в YOLO формат:
    frame000060.txt
    --------------
0 1115 551 93 34
0 1582 127 65 25

    train_ocs.txt
    --------------
data/obj_train_data/ocr/frame000060.txt
"""

import argparse
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument('ocr_folder', help='path to ocr folder')
args = parser.parse_args()

PATH = '/media/max/Transcend/max/plate_recognition/plate_detection_external_datasets/data/ocr_yolo/data'

def find_bbox_coord(data: list, bbox_pattern='position_plate:') -> list:
    """
    Находит координаты боксов в файлах аннотаций датасета OCR,
    затем заменяет специфичный паттерн названия координат бокса на
    паттерн, специфичный для YOLO формата
    """
    yolo_annotation = []
    for line in data:
        match = re.search(bbox_pattern, line)
        if match:
            yolo_style_anat = re.sub(bbox_pattern, '0', line)
            yolo_annotation.append(yolo_style_anat)
    return yolo_annotation

def read_txt_file(text_file: str) -> str:
    """Читает текстовый файл"""
    with open(text_file, 'r') as f:
        data = f.readlines()
    return data

def write_anat_file(text_file: str, anat: str) -> None:
    """Записывает данные аннотаций в текстовый файл """
    with open(text_file, 'w') as f:
        for line in anat:
            f.write(line)

def write_train_file(train_data: list, filename='train_ocr.txt') -> None:
    """Записывает данные о путях к тренировочным данным в отдельный файл"""
    with open(filename, 'w') as train_file:
        for train_path in train_data:
            train_file.write("%s\n" % train_path)

def reannotate2yolostyle(ocr_dataset: str) -> None:
    """Проводит переаннотацию из формата OCR в формат YOLO"""
    train_data = []
    for text_file in pathlib.Path(ocr_dataset).glob('*.txt'):
        data = read_txt_file(text_file)
        anat = find_bbox_coord(data, bbox_pattern='position_plate:')
        write_anat_file(text_file, anat)

        train_data.append('data/obj_train_data/ocr/' + str(text_file.name))
    write_train_file(train_data)

def main():
    reannotate2yolostyle(PATH)

if __name__ == '__main__':
    main()

