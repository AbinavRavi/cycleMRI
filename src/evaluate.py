from metrics.metric import calculate_psnr, calculate_ssim
from network.model import Generator  # noqa
from dataloader.dataloader import loader_instance
from utils.utils import read_config, get_device
from os.path import abspath
from os.path import dirname as d
import torch
import numpy as np


class Inference:
    def __init__(self, config_file):
        config = read_config(config_file)
        self.data_path = config["inference"]["data"]
        self.label_path = config["inference"]["label"]
        self.image_size = config["inference"]["image_size"]
        self.dataloader = loader_instance(
            self.data_path, self.label_path, self.image_size, batch_size=1
        )
        model_path = config["inference"]["model_path"]
        self.model = torch.load(model_path)

    def calculate_metrics_sample(self, image, label):
        # mse = calculate_MSE(label, image)
        ssim = calculate_ssim(label, image)
        psnr = calculate_psnr(label, image)
        return ssim, psnr

    def calculate_metrics_batch(self):
        ssim_arr = []
        psnr_arr = []
        mse_arr = []
        for data, label in self.dataloader:
            data = data.to(torch.float32)
            data = data.to(get_device())
            output_image = self.model(data)
            output_image = output_image.to("cpu")
            # print(output_image, label)
            try:
                ssim, psnr = self.calculate_metrics_sample(
                    output_image.detach().numpy(), label.detach().numpy()
                )
            except Exception:
                print("couldnt calculate metrics")
                ssim = 0
                psnr = 0
            ssim_arr.append(ssim)
            psnr_arr.append(psnr)
            # mse_arr.append(mse)
        mean_mse = np.mean(np.array(mse_arr))
        mean_psnr = np.mean(np.array(psnr))
        mean_ssim = np.mean(np.array(ssim_arr))
        return mean_mse, mean_psnr, mean_ssim

    def generate_t2_image(self, t1c_image):
        t2_image = self.model(t1c_image)
        return t2_image


if __name__ == "__main__":
    parent_dir = f"{d(d(abspath(__file__)))}"
    config_file = parent_dir + "/src/config/config.yml"
    inference = Inference(config_file)
    mean_mse, mean_psnr, mean_ssim = inference.calculate_metrics_batch()
