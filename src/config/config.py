from yacs.config import CfgNode as CN
import os.path as osp

_C = CN()

# Root dir (different for PC and Server)
_C.ROOT = CN()
_C.ROOT.PATH = '/media/max/Transcend/max/plate_recognition/licence_plate_recognition/'

# Chars parameters
_C.CHARS = CN()
_C.CHARS.LIST = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

# LPRNet
_C.LPRNet = CN()
_C.LPRNet.DROPOUT = 0.5
_C.LPRNet.OUT_INDEXES = (2, 6, 13, 22)
_C.LPRNet.PREDICTED_LENGTHS = 18

# Train LPRNet
_C.LPRNet.TRAIN = CN()
_C.LPRNet.TRAIN.OUT_FOLDER = osp.join(_C.ROOT.PATH, 'reports')
_C.LPRNet.TRAIN.PRETRAINED_MODEL = None
_C.LPRNet.TRAIN.PRETRAINED_SPATIAL_TRANSFORMER = None
_C.LPRNet.TRAIN.IMG_SIZE = (94, 24)
_C.LPRNet.TRAIN.BATCH_SIZE = 128
_C.LPRNet.TRAIN.NUM_WORKERS = 4
_C.LPRNet.TRAIN.NUM_EPOCHS = 4

# Plate recognition dataset
_C.LPR_dataset = CN()
_C.LPR_dataset.PATH = osp.join(_C.ROOT.PATH, 'data/raw/licence_recognition/autoriaNumberplateOcrRu-2021-09-01/')
_C.LPR_dataset.TRAIN_PATH = osp.join(_C.LPR_dataset.PATH, 'train')
_C.LPR_dataset.VAL_PATH = osp.join(_C.LPR_dataset.PATH, 'val')
_C.LPR_dataset.TEST_PATH = osp.join(_C.LPR_dataset.PATH, 'test')
_C.LPR_dataset.IMG_SIZE = (94, 24)

def get_cfg_defaults():
    """Get the yacs CfgNode object with default values"""
    return _C.clone()


def combine_config(cfg_path: str):
     """Combine base config with experiment relative config

     Args:
          cfg_path (str): file in .yaml or .yml format
     """
     base_config = get_cfg_defaults()
     if cfg_path is not None and osp.exists(cfg_path):
          base_config.merge_from_file(cfg_path)
     return base_config

