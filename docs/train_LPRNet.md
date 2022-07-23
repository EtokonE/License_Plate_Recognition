## Обучение License Plate Recognition и Spatial transformer layer

### 1. Определить конфигурационный файл
**`src/config/experiment.yaml`**
```yaml
LPRNet:
  TRAIN:
    NUM_EPOCHS: 100
    SAVE_PERIOD: 20
    BATCH_SIZE: 75
    LR: 0.002
    TRAIN_STN: True
    LR_SHED_GAMMA: 0.95
    PRETRAINED_MODEL: ./reports/exp6/weights/lprnet_BEST_model.ckpt
```

**Доступные поля можно найти в базовом конфигурационном файле**
`src/config/config.py`
```python
_C = CN()

# Root dir (different for PC and Server)
_C.ROOT = CN()
_C.ROOT.PATH = '.'  # LPR project dir

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
_C.LPRNet.DROPOUT = 0.5  # Dropout probability
_C.LPRNet.OUT_INDEXES = (2, 6, 13, 22)  # Indices of layers, where we want to extract feature maps and use it
                                        # for embedding in global context
_C.LPRNet.PREDICTED_LENGTHS = 18  # Predicted sequence length

# Train LPRNet
_C.LPRNet.TRAIN = CN()
_C.LPRNet.TRAIN.OUT_FOLDER_NAME = 'reports'  # Folder to save train results
_C.LPRNet.TRAIN.OUT_FOLDER = osp.join(_C.ROOT.PATH, _C.LPRNet.TRAIN.OUT_FOLDER_NAME)
_C.LPRNet.TRAIN.PRETRAINED_MODEL = None  # Pretrained LPR model (.pth)
_C.LPRNet.TRAIN.PRETRAINED_SPATIAL_TRANSFORMER = None  # Pretrained Spatial Transformer model (.pth)
_C.LPRNet.TRAIN.IMG_SIZE = (94, 24)  # Input image size
_C.LPRNet.TRAIN.BATCH_SIZE = 128  # Batch size
_C.LPRNet.TRAIN.NUM_WORKERS = 4  # Num worker
_C.LPRNet.TRAIN.NUM_EPOCHS = 4  # Number of training epochs
_C.LPRNet.TRAIN.SAVE_PERIOD = 2  # How often to save model weights (epoch)
_C.LPRNet.TRAIN.LR = 0.01  # Initial train learning rate
_C.LPRNet.TRAIN.LR_SHED_GAMMA = 0.99  # Gamma for ExponentialLR sheduler
_C.LPRNet.TRAIN.MIN_LR = 0.0001  # Min Learning rate
_C.LPRNet.TRAIN.CTC_REDUCTION = 'mean'  # Reduction: 'none' | 'mean' | 'sum'
_C.LPRNet.TRAIN.TRAIN_STN = True
_C.LPRNet.TRAIN.AVALIABLE_LEN_FOR_COUNTRY = [8, 9]

# Plate recognition dataset
_C.LPR_dataset = CN()
_C.LPR_dataset.PATH_RELATED_ROOT = 'data/raw/recognition_dataset/'  # Train path
_C.LPR_dataset.PATH = osp.join(_C.ROOT.PATH, _C.LPR_dataset.PATH_RELATED_ROOT)
_C.LPR_dataset.TRAIN_FOLDER = 'train'  # Train folder (contains: -> ann & img <- filders)
_C.LPR_dataset.TRAIN_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.TRAIN_FOLDER)
_C.LPR_dataset.VAL_FOLDER = 'val'  # Val folder (contains: -> ann & img <- filders)
_C.LPR_dataset.VAL_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.VAL_FOLDER)
_C.LPR_dataset.TEST_FOLDER = 'test'  # Test folder (contains: -> ann & img <- filders)
_C.LPR_dataset.TEST_PATH = osp.join(_C.LPR_dataset.PATH, _C.LPR_dataset.TEST_FOLDER)
_C.LPR_dataset.IMG_SIZE = (94, 24)  # Standard image size
```

### 2. Структура обучающего [датасета](docs/prepare_lpr_dataset.md)
```tree
├──train
    ├── ann
        ├──A645BH199.json
        ├──B857HP76.json
        ├──...
    ├── img
        ├──A645BH199.png
        ├──B857HP76.png
        ├──...
├──val
    ├──...
├──test
    ├──...
```

### 3. Запуск обучения
```bash
$ python LPR_train.py

ptoinal arguments:
 --out_dir                         Directory to save results
 --config                          Path to experiment configuration
```