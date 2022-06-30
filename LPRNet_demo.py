from src.visualization.tools import convert_output_image, add_text2image, TextPosition
import argparse
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer
from src.config.config import combine_config, get_cfg_defaults
import torch
import time
import cv2
import numpy as np
from src.tools.utils import colorstr, decode_function
from LPR_train import load_weights
from PIL import Image, ImageFont, ImageDraw

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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
    load_weights(model=lprnet, weights='./models/lprnet_BEST_model.ckpt', device=device)
    lprnet.eval()

    STN = SpatialTransformer()
    STN.to(device)
    load_weights(model=STN, weights='./models/stn_BEST_model.ckpt', device=device)
    STN.eval()

    print("Successful to build network!")

    since = time.time()
    image = cv2.imread(args.image)
    im = cv2.resize(image, cfg.LPRNet.TRAIN.IMG_SIZE, interpolation=cv2.INTER_CUBIC)

    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])
    #print(data.shape, 'H' * 10)
    transfer = STN(data)
    #print(transfer, '+'* 10)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 68, 18)
    print(preds.shape)
    labels, pred_labels = decode_function(preds, cfg.CHARS.LIST)
    print("model inference in {:2.3f} seconds".format(time.time() - since))
    print(image.shape, labels, pred_labels)

    #img = cv2ImgAddText(image, labels[0], (0, 0))
    image = add_text2image(image, labels[0], TextPosition(0, 0))

    transformed_img = convert_output_image(transfer)
    #cv2.imshow('transformed', transformed_img)

    cv2.imshow("test", image)
    cv2.waitKey()
    cv2.destroyAllWindows()