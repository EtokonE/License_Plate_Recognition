import numpy as np
import torch
from typing import NamedTuple, Tuple
from PIL import Image, ImageFont, ImageDraw
import cv2



class TextPosition(NamedTuple):
    x_top_teft: int
    y_lop_left: int


def convert_output_image(lpr_output: torch.Tensor) -> np.ndarray:
    converted_lpr_output = lpr_output.squeeze(0).cpu()
    converted_lpr_output = converted_lpr_output.detach().numpy().transpose((1, 2, 0))
    converted_lpr_output = converted_lpr_output.astype('float32')
    converted_lpr_output = 127.5 + converted_lpr_output * 128.
    converted_lpr_output = converted_lpr_output.astype('uint8')
    return converted_lpr_output


def add_text2image(image: np.ndarray,
                   text: str,
                   pos: TextPosition,
                   fill: Tuple = (0, 0, 0),
                   font: str = "data/NotoSansCJK-Regular.ttc",
                   text_size: int = 12) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, text_size, encoding="utf-8")
    draw.text(pos, text, fill, font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




#print(type(convert_output_image(torch.Tensor(3, 24, 94))))
