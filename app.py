# curl -X POST -F file=@09_38_54_8000000_15.png http://localhost:5000/predict
# curl -X POST -F file=@09_38_54_8000000_15.png http://0.0.0.0:8080/predict
import io
import torch
import numpy as np
from PIL import Image
import time
import cv2
from flask import Flask, jsonify, request
from src.config.config import get_cfg_defaults
from src.models.load_functions import load_yolo, load_lprnet, load_stn
from src.models.LPRNet import LPRNet
from src.models.Spatial_transformer import SpatialTransformer
from main import df2json, prepare_detection_output, filter_predictions
from src.tools.utils import decode_function, BeamDecoder


app = Flask(__name__)
cfg = get_cfg_defaults()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Service use device: {device}')
YoloV5 = load_yolo(weights='./models/best.pt', device=device)
LPRNet = load_lprnet(cfg, model=LPRNet, weights='./models/LPRNet_Ep_BEST_model.ckpt', device=device)
STNet = load_stn(cfg, model=SpatialTransformer, weights='./models/SpatialTransformer_Ep_BEST_model.ckpt', device=device)


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = cv2.resize(np.array(image), (1920, 1080), interpolation=cv2.INTER_AREA)
    image = image[:, :, ::-1]
    return image


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    detection = YoloV5(tensor, 1280)
    df_results = detection.pandas().xyxy[0]
    license_plate_batch = prepare_detection_output(df_results, cfg, tensor, device)
    transfer = STNet(license_plate_batch)
    predictions = LPRNet(transfer)
    predictions = predictions.cpu().detach().numpy()

    labels, log_likelihood, pred_labels = decode_function(predictions, cfg.CHARS.LIST, BeamDecoder)
    filtered_predictions = filter_predictions(df_results, labels, log_likelihood)
    df_results['Number'] = filtered_predictions
    return df_results


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        since = time.time()
        file = request.files['file']
        img_bytes = file.read()
        df_results = get_prediction(image_bytes=img_bytes)
        json_results = df2json(df_results)
        print(f'Inference time: {time.time() - since}')
        return jsonify(json_results)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
