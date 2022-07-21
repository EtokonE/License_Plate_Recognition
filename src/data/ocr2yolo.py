import argparse
import pathlib
import re

PATH = '/media/max/Transcend/max/plate_recognition/plate_detection_external_datasets/data/ocr_yolo/data'


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocr_folder', default=PATH, help='path to ocr folder')
    args = parser.parse_args()
    return args

def find_bbox_coord(data: list, bbox_pattern: str = 'position_plate:') -> list:
    """
    Finds the coordinates of boxes in the annotation files of the OCR dataset,
    and replace the specific pattern of the name of the coordinates of the box with a
    pattern specific to the YOLO format
    """
    yolo_annotation = []
    for line in data:
        match = re.search(bbox_pattern, line)
        if match:
            yolo_style_anat = re.sub(bbox_pattern, '0', line)
            yolo_annotation.append(yolo_style_anat)
    return yolo_annotation

def read_txt_file(text_file: str) -> str:
    """Read text file"""
    with open(text_file, 'r') as f:
        data = f.readlines()
    return data

def write_anat_file(text_file: str, anat: str) -> None:
    """Writes annotation data to a text file"""
    with open(text_file, 'w') as f:
        for line in anat:
            f.write(line)

def write_train_file(train_data: list, filename: str = 'train_ocr.txt') -> None:
    """Writes data about training data paths to file"""
    with open(filename, 'w') as train_file:
        for train_path in train_data:
            train_file.write("%s\n" % train_path)

def reannotate2yolostyle(ocr_dataset: str) -> None:
    """Performs reannotation from OCR format to YOLO format"""
    train_data = []
    for text_file in pathlib.Path(ocr_dataset).glob('*.txt'):
        data = read_txt_file(text_file)
        anat = find_bbox_coord(data, bbox_pattern='position_plate:')
        write_anat_file(text_file, anat)

        train_data.append('data/obj_train_data/ocr/' + str(text_file.name))
    write_train_file(train_data)

def main():
    args = create_parser()
    reannotate2yolostyle(args.ocr_folder)

if __name__ == '__main__':
    main()

