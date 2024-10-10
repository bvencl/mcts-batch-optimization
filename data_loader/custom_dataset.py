from abc import abstractmethod
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


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
            self.interpretable_labels = self.labels

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


class MNISTDataset(CustomDataset):
    def __init__(self, data, labels, transform=None, interpretable_labels=None):
        super().__init__(data, labels, transform, interpretable_labels)

    def __getitem__(self, idx):
        img_data = self.data[idx].reshape(28, 28)
        img_data = np.stack([img_data] * 3, axis=-1)  # Convert to 3 channels
        augmented_img = self.transform(Image.fromarray(img_data))
        label = self.labels[idx]

        return augmented_img, label, idx

    def get_original_image(self, idx):
        img_data = self.data[idx].reshape(28, 28)
        img_data = np.stack([img_data] * 3, axis=-1)  # Convert to 3 channels

        return Image.fromarray(img_data.astype(np.uint8))
