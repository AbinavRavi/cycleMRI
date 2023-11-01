from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np


def calculate_psnr(image1, image2):
    """calculate PSNR

    Args:
        image1 (_type_): ground truth image
        image2 (_type_): input image
    """
    psnr_value = peak_signal_noise_ratio(image1, image2)
    return psnr_value


def calculate_ssim(image1, image2):
    """calculate SSIM

    Args:
        image1 (_type_): ground truth image
        image2 (_type_): input image
    """
    ssim_value, _ = structural_similarity(image1, image2)
    return ssim_value


def calculate_MSE(image1, image2):
    """calculate the Mean Squared error between two images

    Args:
        image1 (_type_): image 1
        image2 (_type_): image 2
    """
    mse = np.mean((image1 - image2) ** 2)
    return mse
