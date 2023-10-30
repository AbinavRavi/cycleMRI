from torch.utils.data import Dataset
from utils.utils import to_tensor, resize
from pydicom import dcmread
from glob import glob
import numpy as np
from typing import Tuple


class ImageLoader(Dataset):
    def __init__(self, dataset_path: str, label_path: str, image_size: int) -> None:
        super(ImageLoader, self).__init__()
        self.dataset = glob(f"{dataset_path}/**/**/T1postcontrast/*.dcm")
        self.labels = glob(f"{label_path}/**/**/T2SPACE/*.dcm")
        self.image_size: Tuple = (image_size, image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        pic = dcmread(self.dataset[index]).pixel_array
        label = dcmread(self.labels[index]).pixel_array
        resized_pic = resize(pic, self.image_size)
        resized_label = resize(label, self.image_size)
        normalized_pic = np.expand_dims(resized_pic, axis=-1)
        label = np.expand_dims(resized_label, axis=-1)
        data = to_tensor(normalized_pic)
        label = to_tensor(label)
        return data, label
