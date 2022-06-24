import json
import os.path as osp
from pydantic import BaseModel, ValidationError, Field
from config.config import get_cfg_defaults


Path = str


class ImageSize(BaseModel):
    width: int
    height: int


class Annotation(BaseModel):
    license_plate_number: str = Field(alias='description')
    image_name: str = Field(alias='name')
    image_size: ImageSize = Field(alias='size')


class LPRAnnotationParser:
    """Interface for annotation parsing"""
    def parse_annotation(self, ann_file: Path) -> Annotation:
        raise NotImplementedError


class JsonParser(LPRAnnotationParser):
    """Parse licence plate annotation json file"""
    def _load_json(self, file_path: Path) -> dict:
        try:
            with open(file_path) as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            return ''

    def parse_annotation(self, file_path: Path) -> Annotation:
        json_data = self._load_json(file_path)
        try:
            return Annotation(**json_data)
        except ValidationError as e:
            print('Exception', e.json())


def parse_lpr_annotation(ann_file: Path, parser: LPRAnnotationParser=JsonParser()) -> Annotation:
    """Parse ann_file using parser"""
    return parser.parse_annotation(ann_file)


def main():
    cfg = get_cfg_defaults()
    json_ann_file = osp.join(cfg.LPR_dataset.TRAIN_PATH, 'ann/A001EH62.json')
    lpr_annotation = parse_lpr_annotation(ann_file=json_ann_file,
                                          parser=JsonParser())
    print(lpr_annotation)


if __name__ == '__main__':
    main()
