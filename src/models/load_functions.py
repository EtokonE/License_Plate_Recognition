import torch
from LPR_train import load_weights
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer


def load_yolo(weights='./models/best.pt', device=torch.device('cpu')):
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    yolo.conf = 0.57
    yolo.to(device)
    yolo.eval()
    return yolo


def load_lprnet(cfg, model=LPRNet, weights='./models/LPRNet_Ep_BEST_model.ckpt', device='cpu'):
    lprnet = model(class_num=len(cfg.CHARS.LIST),
                   dropout_prob=0,
                   out_indices=cfg.LPRNet.OUT_INDEXES)
    load_weights(model=lprnet, weights=weights, device=device)
    lprnet.to(device)
    lprnet.eval()
    return lprnet


def load_stn(model=SpatialTransformer, weights='./models/SpatialTransformer_Ep_BEST_model.ckpt', device='cpu'):
    lprnet = model()
    load_weights(model=lprnet, weights=weights, device=device)
    lprnet.to(device)
    lprnet.eval()
    return lprnet
