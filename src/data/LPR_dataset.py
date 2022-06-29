import numpy as np
import cv2
import torch
import os.path as osp
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from src.data.LPR_annotation_parser import Annotation, parse_lpr_annotation
from src.config.config import get_cfg_defaults


class LPRDataset(Dataset):
    """Licence Plate Recognition dataset
    Args:
        data_dir (Path): path to folder with data. Must contains subfolders: 'ann' and 'img'
                         Folder ann contains json annotation files for each image
                         Folder img contains images in .png format
        chars (list):  list containing all possible symbols for a specific country,
                       and additional character '-'
        mode (str): dataset mode, should be 'train', 'val' or 'test'
        img_size (tuple): The final size to which all images will be reduced

    """
    data_modes = ['train', 'val', 'test']

    def __init__(self,
                 data_dir: str,
                 chars: list,
                 mode: str,
                 img_size: Tuple[int, int]):

        self.data_dir = data_dir
        self.annotation_files = list((Path(data_dir) / 'ann').glob('*.json'))
        self.image_dir = osp.join(data_dir, 'img')
        self.chars_dict = {char: i for i, char in enumerate(chars)}
        self.mode = mode

        if self.mode not in self.data_modes:
            print(f'{self.mode} is not correct; correct modes: {self.data_modes}')
            raise NameError

        self.img_size = img_size

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, index):
        annotation = self._get_annotation(annotation_file=self.annotation_files[index])
        x = self._load_sample(annotation)
        x = self._prepare_sample(x)
        if self.mode == 'test':
            return x
        else:
            y = self._get_label(annotation)
            return x, y, len(y)

    def _get_annotation(self, annotation_file: Path):
        return parse_lpr_annotation(ann_file=annotation_file)

    def _load_sample(self, ann: Annotation):
        try:
            image_path = osp.join(self.image_dir, ann.image_name,) + '.png'
            image = cv2.imread(str(image_path))
            return image
        except Exception as e:
            print(str(e))

    def _prepare_sample(self, image):
        cv2.imwrite('reports/raw_image.png', image)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        return image

    def _get_label(self, ann: Annotation):
        plate_number = ann.license_plate_number
        label = list()
        for char in plate_number:
            if char not in self.chars_dict:
                char = self._translit(char)
            label.append(self.chars_dict[char])
        return label

    def _translit(self, char: str) -> str:
        trans_dict = {'А': 'A', 'В': 'B', 'Е': 'E',
                      'К': 'K', 'М': 'M', 'Н': 'H',
                      'О': 'O', 'Р': 'P', 'С': 'C',
                      'Т': 'T', 'У': 'Y', 'Х': 'X'}
        en_char = trans_dict[char.upper]
        return en_char


def collate_fn(batch):
    images = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        images.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten()
    return (torch.stack(images, 0), torch.from_numpy(labels), lengths)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    dataset = LPRDataset(data_dir=Path(cfg.LPR_dataset.VAL_PATH),
                         chars=cfg.CHARS.LIST,
                         mode='val',
                         img_size=cfg.LPR_dataset.IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                            num_workers=2, collate_fn=collate_fn)
    print(f'data length is {dataset.__len__()}')
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))
        break
    im, label, length = dataset[3]
    print(im.shape, label)


