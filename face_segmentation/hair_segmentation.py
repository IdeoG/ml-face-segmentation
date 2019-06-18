import logging
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from face_segmentation.hair_segmentation_model import DeepLabv3_plus
from face_segmentation.image_utils import prepare_nn_image

logger = logging.getLogger(__name__)

_hair_model_path = 'models/deeplab_N5_750_17.pth'
_hair_model: torch.nn.Module = None


def prepare_model(model_dir_path: str, is_root_dir=True):
    """ Load model from checkpoint
    :param model_dir_path:
    :param is_root_dir: If True, then path is the path ROOT_DIR of the module
    :return:
    """
    global _hair_model
    
    logger.debug(f"prepare_model: Started to load model.")
    state_dict_path = f"{model_dir_path}/{_hair_model_path}" if is_root_dir else model_dir_path
    _hair_model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True, _print=False)
    _hair_model.load_state_dict(torch.load(state_dict_path))
    _hair_model.eval()
    if torch.cuda.is_available():
        _hair_model.cuda()

    logger.debug(f"base: model loaded with path = {state_dict_path}")


def inference(image: Union[Image.Image, np.ndarray], mode='pil', hair_threshold=0.5) -> np.ndarray:
    """ Interfere method for hair segmentation

    :param image: Input image with (width = height) and shape (width, width, 3) for example
    :param mode: mode of image
    :param hair_threshold: Binary threshold for each mask channel [0,1]
    :return: mask
    """
    # TODO: In some cases we get holes in area and we should convert area to connected one.
    #       In some cases we get 3 different areas and we should pick area with higher surface
    inputs = prepare_nn_image(image, mode, _hair_model, input_size=256, require_same_dims=False)
    outputs = _hair_model.forward(inputs)

    preds = outputs > hair_threshold
    pred = preds[0][0].cpu().numpy()

    if mode == 'pil':
        mask = cv2.resize(pred, dsize=image.size).astype(bool)
    else:
        w, h = image.shape[:-1]
        mask = cv2.resize(pred, dsize=(h, w)).astype(bool)

    return mask
