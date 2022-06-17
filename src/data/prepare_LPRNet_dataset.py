import json
import os
import sys
from pydantic import BaseModel, ValidationError, Field
sys.path.append('../../')
from config.config import get_cfg_defaults


class ImageSize(BaseModel):
    width: int
    height: int


class Annotation(BaseModel):
    license_plate_number: str = Field(alias='description')
    image_name: str = Field(alias='name')
    image_size: ImageSize = Field(alias='size')


def load_json(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data


def parse_json_data(json_data: dict) -> Annotation:
    try:
        return Annotation(**json_data)
    except ValidationError as e:
        print('Exception', e.json())


def main():
    cfg = get_cfg_defaults()
    json_data = load_json(os.path.join(cfg.LPR_dataset.TRAIN_PATH, 'ann/A001EH62.json'))
    lpr_annotation = parse_json_data(json_data)
    print(lpr_annotation)


if __name__ == '__main__':
    main()
