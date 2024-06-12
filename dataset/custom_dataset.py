from abc import abstractmethod
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, **kwargs):
        assert len(data) == len(labels), "All input lists/tensors must be of the same length!"

        # Ensure data is a NumPy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Ensure labels is a NumPy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Shuffle the data and labels in unison
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        self.data = data[indices]
        self.labels = labels[indices]

        self.transform = transform
        if interpretable_labels is not None:
            self.interpretable_labels = interpretable_labels
        else:
            self.interpretable_labels = labels

        if transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        """Tailor to dataset"""

    @abstractmethod
    def get_original_image(self, idx):
        """Get the original image without any transformations."""


class CifarDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None):
        super().__init__(data, labels, transform, interpretable_labels)

    def __getitem__(self, idx):
        img_data = self.data[idx].reshape(3, 32, 32).transpose((1, 2, 0))
        augmented_img = self.transform(Image.fromarray(img_data))
        label = self.labels[idx]

        return augmented_img, label, idx

    def get_original_image(self, idx):
        """Get the original image without any transformations."""
        # Convert the CIFAR data format to a display-ready format
        img_data = self.data[idx].reshape(3, 32, 32).transpose((2, 1, 0))

        # Convert numpy array to a PIL Image and return
        return Image.fromarray(img_data.astype(np.uint8))


class ImagenetDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, one_hot_encoding=False):
        super().__init__(data, labels, transform, interpretable_labels)
        num_classes = 1000  # Imagenet has 1000 classes
        if one_hot_encoding:
            labels_tensor = torch.tensor(self.labels, dtype=torch.int64)
            self.labels = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] - 1

        return img, label, idx

    def get_original_image(self, idx):
        """Get the original image without any transformations."""
        return Image.open(self.data[idx]).convert('RGB')


class TinyImagenetDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None, one_hot_encoding=True):
        super().__init__(data, labels, transform, interpretable_labels, num_classes=200)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]

        return img, label, idx

    def get_original_image(self, idx):
        return Image.open(self.data[idx]).convert('RGB')