import pandas as pd
import torch
import cv2
import time
import argparse
import json
import os
import numpy as np
from pathlib import Path
from src.config.config import get_cfg_defaults
from src.data.coco2lpr import BboxXYXY
from src.tools.utils import decode_function, BeamDecoder
from src.visualization.tools import add_text2image, TextPosition
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer
from src.models.load_functions import load_yolo, load_lprnet, load_stn



def create_parser():
    parser = argparse.ArgumentParser(description='Parameters to inference LPR pipeline')
    parser.add_argument('--yolo_weights', type=str,
                        default='./models/best.pt',
                        help='Pretrained yolov5 weights')
    parser.add_argument('--lprnet_weights', type=str,
                        default='./models/LPRNet_Ep_BEST_model.ckpt',
                        help='Pretrained LPRNet weights')
    parser.add_argument('--stn_weights', type=str,
                        default='./models/SpatialTransformer_Ep_BEST_model.ckpt',
                        help='Pretrained STNet weights')
    parser.add_argument('--image', type=str,
                        default='reports/figures/yolo/09_38_54_8000000_15.png',
                        help='Path to image')
    parser.add_argument('--out_dir', type=str,
                        default='reports/',
                        help='Path to directory where would save results')
    parser.add_argument('--save_json', type=bool,
                        default=True,
                        help='Save JSON with results')
    parser.add_argument('--save_img', type=bool,
                        default=True,
                        help='Save image with results')
    parser.add_argument('--save_dataframe', type=bool,
                        default=True,
                        help='Save dataframe with results')

    args = parser.parse_args()
    return args


def prepare_detection_input(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1]
    return img


def prepare_detection_output(df_results: pd.DataFrame, cfg, img, device) -> torch.Tensor:
    cropped_images = []
    for row in range(df_results.shape[0]):
        height = df_results.iloc[row][3] - df_results.iloc[row][1]
        curr_bbox = BboxXYXY(
            top_x=int(df_results.iloc[row][0]),
            top_y=int(df_results.iloc[row][1] - height * 0.2),
            bottom_x=int(df_results.iloc[row][2]),
            bottom_y=int(df_results.iloc[row][3] + height * 0.2)
        )

        license_plate = img[curr_bbox.top_y:curr_bbox.bottom_y, curr_bbox.top_x:curr_bbox.bottom_x]
        license_plate = cv2.resize(license_plate, cfg.LPRNet.TRAIN.IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        license_plate = (np.transpose(np.float32(license_plate), (2, 0, 1)) - 127.5) * 0.0078125
        cropped_images.append(license_plate)
    return torch.from_numpy(np.array(cropped_images)).float().to(device)


def filter_predictions(df_results, labels, log_likelihood):
    final_labels = []
    for row in range(df_results.shape[0]):
        if (log_likelihood[row] < -85) and (len(labels[row]) in [8, 9]):
            final_labels.append(labels[row])
        else:
            final_labels.append(None)
    return final_labels


def df2json(df_results):
    return df_results.to_json(orient="index")


def create_out_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def save_json(out_dir, json_dict, source=''):
    with open(os.path.join(out_dir, f'{source}_results.json'), 'w', encoding='utf8') as f:
        json.dump(json_dict, f)


def save_dataframe(out_dir, df_results, source=''):
    df_results.to_csv(os.path.join(out_dir, f'{source}_results.csv'))


def save_img(out_dir, yolo_results, df_results, source=''):
    image = yolo_results.render()[0]
    for row in range(df_results.shape[0]):
        if df_results.iloc[row][7] != None:
            image = add_text2image(image, str(df_results.iloc[row][7]),
                                   TextPosition(df_results.iloc[row][0], df_results.iloc[row][3]),
                                   text_size=40)

    cv2.imwrite(os.path.join(out_dir, f'{source}2_results.png'), image)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    args = create_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create models
    YoloV5 = load_yolo(weights=args.yolo_weights, device=device)
    LPRNet = load_lprnet(cfg, model=LPRNet, weights=args.lprnet_weights, device=device)
    STNet = load_stn(cfg, model=SpatialTransformer, weights=args.stn_weights, device=device)
    # Detection
    since = time.time()
    img = prepare_detection_input(args.image)
    detection = YoloV5(img, size=1280)
    df_results = detection.pandas().xyxy[0]
    # Recognition
    license_plate_batch = prepare_detection_output(df_results, cfg, img, device)
    transfer = STNet(license_plate_batch)
    predictions = LPRNet(transfer)
    predictions = predictions.cpu().detach().numpy()
    # Postprocess
    labels, log_likelihood, pred_labels = decode_function(predictions, cfg.CHARS.LIST, BeamDecoder)
    filtered_predictions = filter_predictions(df_results, labels, log_likelihood)
    df_results['Number'] = filtered_predictions
    # Save results
    if args.save_json or args.save_img or args.save_dataframe:
        create_out_dir(out_dir=args.out_dir)
        source = Path(args.image).stem

    if args.save_json:
        json_dict = df2json(df_results)
        save_json(args.out_dir, json_dict, source)

    if args.save_dataframe:
        save_dataframe(args.out_dir, df_results, source)

    if args.save_img:
        save_img(args.out_dir, detection, df_results, source)
