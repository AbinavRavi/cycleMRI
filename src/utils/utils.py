import numpy as np
import torch
import yaml
import sklearn.preprocessing as skp
from skimage import transform


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def to_tensor(array):
    """
    input image array of form WxHxD
    output tensor of the form DxHxW
    """
    img = np.transpose(array, (2, 0, 1))
    img = torch.from_numpy(img)
    return img


def normalize(array):
    """
    Input image is given as a 2D array
    Resizing is done with sklearn.preprocessing.normalize
    """
    image = skp.normalize(array[:, :, 0])
    return image


def resize(array, size):
    """Resizes a given array into size shape

    Args:
        array (numpy.ndarray): Image to be resized
        size (Tuple): Size of the output image
    """
    image = transform.resize(array, size)
    return image


def read_config(file):
    with open(file, "r") as stream:
        cxr_config = yaml.full_load(stream)
    return cxr_config
