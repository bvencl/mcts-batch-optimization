from torchvision import transforms
import _pickle as pickle
import sys
import gzip
from utils.utils import *


def load_mnist(config, data_dir):
    file_name = "mnist.pkl.gz"
    path = os.path.join(data_dir, file_name)

    if os.path.exists(path):
        f = gzip.open(path, "rb")
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding="bytes")
        f.close()
    else:
        raise ValueError(
            f"MNIST data not found in {data_dir}. Make sure to download mnist.pkl.gz."
        )

    (train_data, train_labels), (val_data, val_labels) = data
    all_data = np.concatenate((train_data, val_data))
    all_labels = np.concatenate((train_labels, val_labels))

    num_val_test_samples = config.data_loader.num_data_samples_val_test
    val_test_split = config.data_loader.val_test_split

    val_test_data = all_data[:num_val_test_samples]
    val_test_labels = all_labels[:num_val_test_samples]

    train_data = all_data[num_val_test_samples:]
    train_labels = all_labels[num_val_test_samples:]

    num_test_samples = int(num_val_test_samples * val_test_split)
    num_val_samples = num_val_test_samples - num_test_samples

    val_data = val_test_data[:num_val_samples]
    val_labels = val_test_labels[:num_val_samples]

    test_data = val_test_data[num_val_samples:]
    test_labels = val_test_labels[num_val_samples:]

    train_labels, val_labels, test_labels = (
        train_labels.tolist(),
        val_labels.tolist(),
        test_labels.tolist(),
    )
    train_labels = [int(train_item) for train_item in train_labels]
    val_labels = [int(val_item) for val_item in val_labels]
    test_labels = [int(test_item) for test_item in test_labels]

    if len(test_labels) == 0:
        print("No test dataset is provided. Using only validation data.")
    print(
        f"Number of validation labels: {len(val_labels)}, number of test labels: {len(test_labels)}"
    )
    interpretable_labels = None

    testing_data = False if len(test_labels) == 0 else True

    resize_param = 28
    mean = np.mean(train_data / 255.0)
    std = np.std(train_data / 255.0)

    train_transforms = []
    val_transforms = []
    test_transforms = []

    if config.data_loader.augmentation:
        train_transforms_temp = [
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=(resize_param, resize_param), scale=(0.75, 0.95)
                    )
                ],
                p=0.8,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                    )
                ],
                p=0.75,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.5
            ),
        ]
        train_transforms.append(
            transforms.RandomApply(transforms=train_transforms_temp, p=0.9)
        )

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    train_transforms.append(transforms.Normalize(mean=(mean,), std=(std,)))
    val_transforms.append(transforms.Normalize(mean=(mean,), std=(std,)))
    test_transforms.append(transforms.Normalize(mean=(mean,), std=(std,)))

    train_transforms = transforms.Compose(transforms=train_transforms)
    val_transforms = transforms.Compose(transforms=val_transforms)
    test_transforms = transforms.Compose(transforms=test_transforms)

    num_data_samples_train = int(config.data_loader.num_data_samples_train)
    num_data_samples_val = int(len(val_labels))
    num_data_samples_test = int(len(test_labels))
    num_data_samples_mcts = int(config.data_loader.num_data_samples_mcts)

    mnist_data = dict()

    if config.data_loader.custom_sampler:
        mnist_data["train_data"] = train_data[:num_data_samples_mcts]
        mnist_data["train_labels"] = train_labels[:num_data_samples_mcts]
        mnist_data["val_data"] = val_data[:num_data_samples_val]
        mnist_data["val_labels"] = val_labels[:num_data_samples_val]
        mnist_data["test_data"] = test_data[:num_data_samples_test]
        mnist_data["test_labels"] = test_labels[:num_data_samples_test]
        print(f"Number of mcts training labels: {len(mnist_data['train_labels'])}")
    else:
        mnist_data["train_data"] = train_data[:num_data_samples_train]
        mnist_data["train_labels"] = train_labels[:num_data_samples_train]
        mnist_data["val_data"] = val_data[:num_data_samples_val]
        mnist_data["val_labels"] = val_labels[:num_data_samples_val]
        mnist_data["test_data"] = test_data[:num_data_samples_test]
        mnist_data["test_labels"] = test_labels[:num_data_samples_test]
        print(f"Number of training labels: {len(mnist_data['train_labels'])}")

    mnist_data["train_transforms"] = train_transforms
    mnist_data["val_transforms"] = val_transforms
    mnist_data["test_transforms"] = val_transforms
    mnist_data["interpretable_labels"] = interpretable_labels

    return mnist_data
