from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def expand_image_shape(image: Union[Image.Image, np.ndarray], n_channels=3, mode='pil') -> Union[Image.Image, np.array]:
    """ Add black areas to expand image shape and make it equal (width = height)
    :param image:
    :param n_channels:
    :param mode: Supported modes = 'pil', 'rgb', 'bgr'
    :return:
    """

    copied_image = np.array(image)

    if mode not in ['pil', 'rgb', 'bgr']:
        raise ValueError(f"Supported modes: 'pil', 'rgb', 'bgr'. Got mode '{mode}' and type '{type(image)}'")

    w, h = copied_image.shape[:-1] if n_channels == 3 else copied_image.shape
    if w == h:
        return image

    minor_bound = h if h < w else w
    major_bound = w if w > h else h

    axis = 0 if w < h else 1

    left_diff_bound = (major_bound - minor_bound) // 2
    right_diff_bound = (major_bound - minor_bound) - left_diff_bound

    expanded_image = np.insert(copied_image, [0 for _ in range(left_diff_bound)], 0, axis=axis)
    expanded_image = np.insert(expanded_image, [minor_bound + left_diff_bound for _ in range(right_diff_bound)], 0,
                               axis=axis)

    return Image.fromarray(expanded_image) if mode == 'pil' else expanded_image


def prepare_nn_image(image: Union[Image.Image, np.ndarray], mode: str,
                     model: torch.nn.Module, input_size=224, require_same_dims=False) -> torch.tensor:
    if mode == 'bgr' and isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    elif mode == 'rgb' and isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif mode == 'pil' and isinstance(image, Image.Image):
        pass
    else:
        raise ValueError(f"Supported modes: 'pil', 'rgb', 'bgr'. Got mode '{mode}' and type '{type(image)}'")

    width, height = image.size
    if require_same_dims:
        if width != height:
            raise ValueError(f"Width and height doesnt match {image.size}")

    if model is None:
        raise ValueError("Model didnt loaded. Please use method 'prepare_model'")

    output_size = width
    input_tfms, _ = _inference_transforms(input_size, output_size)
    inputs = input_tfms(image).unsqueeze(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    return inputs


def _inference_transforms(input_size=256, output_size=256):
    input_tfms = transforms.Compose([transforms.Resize(input_size),
                                     transforms.ToTensor()])
    output_tfms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(output_size)])
    return input_tfms, output_tfms