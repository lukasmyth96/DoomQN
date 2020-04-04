import numpy as np
from skimage import transform
from skimage.color import rgb2gray


def preprocess(img, resolution, convert_to_gray):
    """
    resize img and convert to grayscale if convert_to_gray=True
    Parameters
    ----------
    img: np.ndarray
    resolution: tuple
        (w, h)
    convert_to_gray: bool

    Returns
    -------
    preprocessed_img
    """
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)  # convert channels first to channels last format
    preprocessed_img = transform.resize(img, resolution)
    if convert_to_gray:
        preprocessed_img = rgb2gray(preprocessed_img)
        preprocessed_img = np.expand_dims(preprocessed_img, axis=-1)  # add channel dim of 1 - needed for model

    return preprocessed_img


