"""
This script converts the labelling of the coco format into a dataset format for license plate recognition
├──train
    ├── ann
        ├──A645BH199.json
        ├──B857HP76.json
        ├──...
    ├── img
        ├──A645BH199.png
        ├──B857HP76.json
        ├──...
├──val
    ├──...
├──test
    ├──...
"""
import random
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel, ValidationError, Field
from typing import List, NamedTuple, Optional
from src.data.LPR_annotation_parser import ImageSize
from src.config.config import get_cfg_defaults


cfg = get_cfg_defaults()
AVAILABLE_CHARS = cfg.CHARS.LIST
AVAILABLE_CHARS.append('_')

Image = np.ndarray


class LPRAnnotation(BaseModel):
    description: str
    name: str
    size: ImageSize
    original_image: str


class CocoBbox(NamedTuple):
    top_x: float
    top_y: float
    width: float
    height: float


class CocoAttributes(BaseModel):
    Number: str = None
    rotation: float = None


class CocoImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    area: float
    bbox: CocoBbox
    attributes: CocoAttributes


class CocoDataset(BaseModel):
    images: List[CocoImage] = Field(alias='images')
    annotations: List[CocoAnnotation] = Field(alias='annotations')


class BboxXYXY(NamedTuple):
    top_x: int
    top_y: int
    bottom_x: int
    bottom_y: int


def create_parser():
    parser = argparse.ArgumentParser(description='Arguments to convert coco format to lpr format')
    parser.add_argument('--image_dir', type=str, help='Directory with images')
    parser.add_argument('--coco_json_file', type=str, help='Json file with annotations in json format')
    parser.add_argument('--out_dir', type=str, help='Directory to save results')
    args = parser.parse_args()
    return args


def create_lpr_folder_structure(root_folder: Path) -> None:
    if not isinstance(root_folder, Path):
        root_folder = Path(root_folder)
    root_folder.mkdir(parents=True, exist_ok=True)
    (root_folder / 'ann').mkdir(parents=True, exist_ok=True)
    (root_folder / 'img').mkdir(parents=True, exist_ok=True)


def get_folders_path(root_folder: Path) -> (Path, Path):
    return root_folder / 'ann', root_folder / 'img'


def load_coco_dataset_annotations(coco_file: Path) -> Optional[dict]:
    try:
        with open(coco_file, 'r', encoding='utf8') as f:
            json_data = json.load(f)
        return json_data
    except FileNotFoundError:
        return None


def get_coco_dataset_annotations(json_data: dict) -> CocoDataset:
    try:
        return CocoDataset(**json_data)
    except ValidationError as e:
        print('Exception', e.json())



def get_single_annotation(coco_dataset: CocoDataset, index: int) -> CocoAnnotation:
    return coco_dataset.annotations[index]


def get_single_image_ann(coco_dataset: CocoDataset, index: int) -> CocoImage:
    return coco_dataset.images[index]


def process_fileame(filename: str, available_chars: List) -> str:
    filename = Path(filename).stem
    translited_filename = translit(str(filename), available_chars)
    return translited_filename


def translit(sequence: str, available_chars: List):
    en_chars = []
    trans_dict = {'А': 'A', 'В': 'B', 'Е': 'E',
                  'К': 'K', 'М': 'M', 'Н': 'H',
                  'О': 'O', 'Р': 'P', 'С': 'C',
                  'Т': 'T', 'У': 'Y', 'Х': 'X',
                  '-': '-', '_': '_'}
    for char in str(sequence):
        if char not in available_chars:
            char = trans_dict[char.upper()]
        en_chars.append(char)
    final_str = ''.join(en_chars)
    return final_str


def coco_annotation2lpr_annotation(coco_dataset: CocoDataset, index: int) -> LPRAnnotation:
    annotation = get_single_annotation(coco_dataset, index)
    image_id = annotation.image_id
    image_ann = coco_dataset.images[image_id - 1] if coco_dataset.images[image_id - 1].id == image_id else None
    original_filename = image_ann.file_name

    if len(annotation.attributes.Number) != 0:
        image_filename = process_fileame(annotation.attributes.Number, AVAILABLE_CHARS)
    else:
        image_filename = process_fileame(image_ann.file_name, AVAILABLE_CHARS)

    return LPRAnnotation(
        description = translit(annotation.attributes.Number, AVAILABLE_CHARS),
        name = image_filename + '_' + str(random.randint(1, 650)),
        size = ImageSize(width=annotation.bbox.width,
                               height=annotation.bbox.height),
        original_image = original_filename
    )


def save_lpr_annotation(lpr_annotation: LPRAnnotation, ann_folder: Path) -> None:
    annotation_json = ann_folder / (lpr_annotation.name + '.json')
    with open(annotation_json, 'w', encoding='utf8') as f:
        json.dump(lpr_annotation.dict(), f)


def load_image(image_folder: Path, filename: str) -> Image:
    image_path = image_folder / filename
    image = cv2.imread(str(image_path))
    return image


def get_increased_coordinates(bbox: CocoBbox,
                              multiplier_y: float = 1.5,
                              multiplier_x: float = 1.0) -> BboxXYXY:
    return BboxXYXY(
        top_x = int(bbox.top_x - bbox.width * (multiplier_x - 1)),
        top_y = int(bbox.top_y - bbox.height * (multiplier_y - 1)),
        bottom_x = int(bbox.top_x + bbox.width * multiplier_x),
        bottom_y = int(bbox.top_y + bbox.height * multiplier_y)
    )



def crop_license_plate(image: Image, bbox: CocoBbox) -> Image:
    box2crop = get_increased_coordinates(bbox, multiplier_y=1.35)
    return image[box2crop.top_y:box2crop.bottom_y, box2crop.top_x:box2crop.bottom_x]


def save_cropped_image(image: Image,
                     path: Path,
                     filename: str,
                     extension: str = '.png') -> None:
    full_path = str((path / filename).with_suffix(extension))
    cv2.imwrite(full_path, image)


def main():
    args = create_parser()

    create_lpr_folder_structure(root_folder=Path(args.out_dir))
    ann_path, img_path = get_folders_path(Path(args.out_dir))

    coco_json_data = load_coco_dataset_annotations(coco_file=Path(args.coco_json_file))
    coco_data = get_coco_dataset_annotations(coco_json_data)

    for i in range(len(coco_data.annotations)):
        try:
            lpr = coco_annotation2lpr_annotation(coco_dataset=coco_data, index=i)
            if len(lpr.description) == 0:
                continue
            full_image = load_image(image_folder=Path(args.image_dir), filename=lpr.original_image)
            bbox = crop_license_plate(full_image, coco_data.annotations[i].bbox)
            save_cropped_image(image=bbox,
                               path=img_path,
                               filename=lpr.name,
                               extension='.png')
            save_lpr_annotation(lpr, ann_path)
        except Exception as e:
            print(e)
            continue
        print(lpr)


if __name__ == '__main__':
    main()
