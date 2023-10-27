import numpy as np
import torch
import yaml
import sklearn.preprocessing as skp


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


def read_config(file):
    with open(file, "r") as stream:
        cxr_config = yaml.full_load(stream)
    return cxr_config
