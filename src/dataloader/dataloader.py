from torch.utils.data import Dataset, DataLoader
from utils.utils import to_tensor, resize
from pydicom import dcmread
from glob import glob
import numpy as np
from typing import Tuple
import torch


class ImageLoader(Dataset):
    def __init__(self, dataset_path: str, label_path: str, image_size: int) -> None:
        super(ImageLoader, self).__init__()
        self.dataset = glob(f"{dataset_path}/*.dcm", recursive=True)
        self.labels = glob(f"{label_path}/*.dcm", recursive=True)
        self.image_size: Tuple = (image_size, image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            pic = dcmread(self.dataset[index]).pixel_array
            label = dcmread(self.labels[index]).pixel_array
            resized_pic = resize(pic, self.image_size)
            resized_label = resize(label, self.image_size)
            normalized_pic = np.expand_dims(resized_pic, axis=-1)
            label = np.expand_dims(resized_label, axis=-1)
            data = to_tensor(normalized_pic)
            label = to_tensor(label)
            return data, label
        except Exception:
            print(f"{self.dataset[index]} is the reason why")
            return torch.zeros(1, 128, 128), torch.zeros(1, 128, 128)


def loader_instance(dataset_path, label_path, image_size, batch_size):
    dataset = ImageLoader(dataset_path=dataset_path, label_path=label_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
