from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):

        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            self.imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 1).astype(np.float32) / 255.0 
            # shape: (num_images, rows, cols, channels=1)

        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        self.transforms = transforms

    def __getitem__(self, index) -> object:

        if isinstance(index, int):
            return self.apply_transforms(self.imgs[index]), self.labels[index]
        else:
            return np.stack([self.apply_transforms(img) for img in self.imgs[index]]), self.labels[index]

    def __len__(self) -> int:

        return self.labels.shape[0]
