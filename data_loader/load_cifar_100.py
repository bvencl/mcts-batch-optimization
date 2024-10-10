from torchvision import transforms
import tarfile
from utils.utils import *


def load_cifar_100(config, data_dir):
    file = "cifar-100-python"

    # Path to the extracted CIFAR data
    extracted_path = os.path.join(data_dir, file)

    # Check if the dataset files are already extracted
    if not os.path.exists(os.path.join(extracted_path, 'train')):
        print("Cifar100 loader: Extracting zipped files")
        tar_path = os.path.join(data_dir, file + '.tar.gz')
        if os.path.exists(tar_path):
            # Extract the archive
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
        else:
            raise ValueError(f"Cifar100 data not found in {data_dir}. Make sure to download cifar-x-python.tar.gz.")

    # CIFAR Training File
    print("Cifar100 loader: Load training data")
    train_file = os.path.join(extracted_path, 'train')
    train_batch = unpickle(train_file)
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']  # Note that CIFAR has 'fine_labels' for the detailed classes

    # CIFAR Test File
    print("Cifar100 loader: Load test data")
    test_file = os.path.join(extracted_path, 'test')
    test_batch = unpickle(test_file)
    val_data = test_batch[b'data']
    val_labels = test_batch[b'fine_labels']

    # Label Names
    print("Cifar100 loader: Load labels")
    meta_file = os.path.join(extracted_path, 'meta')
    meta = unpickle(meta_file)
    interpretable_labels = [label.decode('utf-8') for label in
                            meta[b'fine_label_names']]  # Use 'fine_label_names' for CIFAR
    resize_param = 32
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    # Transforms for training
    train_transforms = []
    val_transforms = []

    if config.data_loader.augmentation:
        train_transforms.append(transforms.Resize((resize_param, resize_param)))
        val_transforms.append(transforms.Resize((resize_param, resize_param)))

        train_transforms_temp = [transforms.RandomApply([transforms.RandomResizedCrop(size=(resize_param, resize_param),
                                                                                      scale=(0.75, 0.95))], p=0.8),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                                saturation=0.3, hue=0.15)], p=0.75),
                                 transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5)]
        train_transforms.append(transforms.RandomApply(transforms=train_transforms_temp, p=0.9))

        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())

        train_transforms.append(transforms.Normalize(mean=mean, std=std))
        val_transforms.append(transforms.Normalize(mean=mean, std=std))

    else:
        train_transforms.append(transforms.Resize((resize_param, resize_param)))
        train_transforms.append(transforms.ToTensor())

        val_transforms.append(transforms.Resize((resize_param, resize_param)))
        val_transforms.append(transforms.ToTensor())

    train_transforms = transforms.Compose(transforms=train_transforms)
    val_transforms = transforms.Compose(transforms=val_transforms)

    cifar_data = dict()
    cifar_data["train_data"] = train_data
    cifar_data["train_labels"] = train_labels
    cifar_data["val_data"] = val_data
    cifar_data["val_labels"] = val_labels
    cifar_data["train_transforms"] = train_transforms
    cifar_data["val_transforms"] = val_transforms
    cifar_data["interpretable_labels"] = interpretable_labels

    return cifar_data
