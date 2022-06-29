from yacs.config import CfgNode as CN
import os.path as osp

_C = CN()

# Root dir (different for PC and Server)
_C.ROOT = CN()
#_C.ROOT.PATH = '/media/max/Transcend/max/plate_recognition/licence_plate_recognition/' # LPR project dir
_C.ROOT.PATH = '.'

# Chars parameters
_C.CHARS = CN()
# List of all possible characters
_C.CHARS.LIST = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

# LPRNet
_C.LPRNet = CN()
_C.LPRNet.DROPOUT = 0.5 # Dropout probability
_C.LPRNet.OUT_INDEXES = (2, 6, 13, 22) # Indices of layers, where we want to extract feature maps and use it
                                       # for embedding in global context
_C.LPRNet.PREDICTED_LENGTHS = 18 # Predicted sequence length

# Train LPRNet
_C.LPRNet.TRAIN = CN()
_C.LPRNet.TRAIN.OUT_FOLDER_NAME = 'reports' # Folder to save train results
_C.LPRNet.TRAIN.OUT_FOLDER = osp.join(_C.ROOT.PATH, _C.LPRNet.TRAIN.OUT_FOLDER_NAME)
_C.LPRNet.TRAIN.PRETRAINED_MODEL = None # Pretrained LPR model (.pth)
_C.LPRNet.TRAIN.PRETRAINED_SPATIAL_TRANSFORMER = None # Pretrained Spatial Transformer model (.pth)
_C.LPRNet.TRAIN.IMG_SIZE = (94, 24) # Input image size
_C.LPRNet.TRAIN.BATCH_SIZE = 128 # Batch size
_C.LPRNet.TRAIN.NUM_WORKERS = 4 # Num worker
_C.LPRNet.TRAIN.NUM_EPOCHS = 4 # Number of training epochs
_C.LPRNet.TRAIN.SAVE_PERIOD = 2 # How often to save model weights (epoch)
_C.LPRNet.TRAIN.LR = 0.01 # Initial train learning rate
_C.LPRNet.TRAIN.LR_SHED_GAMMA = 0.99 # Gamma for ExponentialLR sheduler
_C.LPRNet.TRAIN.MIN_LR = 0.0001 # Min Learning rate
_C.LPRNet.TRAIN.CTC_REDUCTION = 'mean' # Reduction: 'none' | 'mean' | 'sum'
_C.LPRNet.TRAIN.TRAIN_STN = True
_C.LPRNet.TRAIN.AVALIABLE_LEN_FOR_COUNTRY = [8, 9]

# Plate recognition dataset
_C.LPR_dataset = CN()
_C.LPR_dataset.PATH_RELATED_ROOT = 'data/raw/recognition_dataset/' # Train path
_C.LPR_dataset.PATH = osp.join(_C.ROOT.PATH, _C.LPR_dataset.PATH_RELATED_ROOT)
_C.LPR_dataset.TRAIN_FOLDER = 'train' # Train folder (contains: -> ann & img <- filders)
_C.LPR_dataset.TRAIN_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.TRAIN_FOLDER)
_C.LPR_dataset.VAL_FOLDER = 'val' # Val folder (contains: -> ann & img <- filders)
_C.LPR_dataset.VAL_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.VAL_FOLDER)
_C.LPR_dataset.TEST_FOLDER = 'test' # Test folder (contains: -> ann & img <- filders)
_C.LPR_dataset.TEST_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.TEST_FOLDER)
_C.LPR_dataset.IMG_SIZE = (94, 24) # Standard image size

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

     base_config.LPR_dataset.PATH = osp.join(base_config.ROOT.PATH, base_config.LPR_dataset.PATH_RELATED_ROOT)
     base_config.LPR_dataset.TRAIN_PATH = osp.join(base_config.LPR_dataset.PATH, base_config.LPR_dataset.TRAIN_FOLDER)
     base_config.LPR_dataset.VAL_PATH = osp.join(base_config.LPR_dataset.PATH, base_config.LPR_dataset.VAL_FOLDER)
     base_config.LPR_dataset.TEST_PATH = osp.join(base_config.LPR_dataset.PATH, base_config.LPR_dataset.TEST_FOLDER)
     return base_config

