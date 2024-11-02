import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        imgs, labels = [], []
        if train: # load training dataset
            filenames = [f'data_batch_{i}' for i in range(1, 6)]
        else: # load test dataset
            filenames = ['test_batch']

        for i, filename in enumerate(filenames):
            with open(os.path.join(base_folder, filename), 'rb') as file:
                data = pickle.load(file, encoding='bytes')
                imgs.append(data[b'data'])
                labels.append(data[b'labels'])
        self.X = np.concatenate(imgs, axis=0).reshape((-1, 3, 32, 32)) / 255.
        self.y = np.concatenate(labels)
               
        # print(self.X.shape, self.y.shape)
        self.p = p
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms is not None:
            imgs = [self.transforms(img) for img in self.X[index]]
        else:
            imgs = self.X[index]
        return imgs, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
