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
     'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т',
     'У', 'Х', '-'
]

# LPRNet
_C.LPRNet = CN()
_C.LPRNet.DROPOUT = 0.5
_C.LPRNet.OUT_INDICEC = (2, 6, 13, 22)

# Plate recognition dataset
_C.LPR_dataset = CN()
_C.LPR_dataset.PATH = osp.join(_C.ROOT.PATH, 'data/raw/licence_recognition/autoriaNumberplateOcrRu-2021-09-01/')
_C.LPR_dataset.TRAIN_PATH = osp.join(_C.LPR_dataset.PATH, 'train')
_C.LPR_dataset.VAL_PATH = osp.join(_C.LPR_dataset.PATH, 'val')
_C.LPR_dataset.TEST_PATH = osp.join(_C.LPR_dataset.PATH, 'test')





def get_cfg_defaults():
    """Get the yacs CfgNode object with default values"""
    return _C.clone()