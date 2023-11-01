from metrics.metric import calculate_MSE, calculate_psnr, calculate_ssim
from network.model import Generator  # noqa
from dataloader.dataloader import loader_instance
from utils.utils import read_config
from os.path import abspath
from os.path import dirname as d
import torch


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
        mse = calculate_MSE(label, image)
        ssim = calculate_ssim(label, image)
        psnr = calculate_psnr(label, image)
        return mse, ssim, psnr

    def calculate_metrics_batch(self):
        pass

    def generate_t2_image(self, t1c_image):
        pass


if __name__ == "__main__":
    parent_dir = f"{d(d(abspath(__file__)))}"
    config_file = parent_dir + "/src/config/config.yml"
    inference = Inference(config_file)
