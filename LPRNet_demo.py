from src.visualization.tools import convert_output_image, add_text2image, TextPosition
import argparse
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer
from src.config.config import combine_config, get_cfg_defaults
import torch
import time
import cv2
import numpy as np
from src.tools.utils import decode_function, BeamDecoder
from LPR_train import load_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("--image", help='image path', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()

    lprnet = LPRNet(class_num=len(cfg.CHARS.LIST),
                    dropout_prob=0,
                    out_indices=cfg.LPRNet.OUT_INDEXES)
    lprnet.to(device)
    load_weights(model=lprnet, weights='./models/LPRNet_Ep_BEST_model.ckpt', device=device)
    lprnet.eval()

    STN = SpatialTransformer()
    STN.to(device)
    load_weights(model=STN, weights='./models/SpatialTransformer_Ep_BEST_model.ckpt', device=device)
    STN.eval()

    print("Successful to build network!")

    since = time.time()
    image = cv2.imread(args.image)
    im = cv2.resize(image, cfg.LPRNet.TRAIN.IMG_SIZE, interpolation=cv2.INTER_CUBIC)

    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])

    transfer = STN(data)
    predictions = lprnet(transfer)
    predictions = predictions.cpu().detach().numpy()  # (1, 68, 18)
    labels, prob, pred_labels = decode_function(predictions, cfg.CHARS.LIST, BeamDecoder)
    print("model inference in {:2.3f} seconds".format(time.time() - since))

    transformed_img = convert_output_image(transfer)
    pad_image = cv2.copyMakeBorder(transformed_img, top=15, bottom=0, left=0, right=0,
                                   borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    if (prob[0] < -85) and (len(labels[0]) in [8, 9]):
        pad_image = add_text2image(pad_image, (labels[0]), TextPosition(16, 0), text_size=10)

    cv2.imshow('Prediction', pad_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(prob[0])
