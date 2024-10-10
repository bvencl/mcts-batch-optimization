import os

from factory.base_factory import BaseFactory
from data_loader.custom_dataset import CifarDataset, MNISTDataset
from data_loader.load_cifar import load_cifar
from data_loader.load_mnist import load_mnist


class DatasetFactory(BaseFactory):

    @classmethod
    def create(cls, **kwargs):
        assert "config" in kwargs, "No config is given"
        data_folder_name = kwargs["config"].data_loader.dataset

        root = os.getcwd()
        file_name = root + "/data/" + data_folder_name + "/"
        if data_folder_name == "cifar100":
            training_dataset = load_cifar(
                config=kwargs["config"], data_dir=file_name, num_classes=100
            )
        elif data_folder_name == "cifar10":
            training_dataset = load_cifar(
                config=kwargs["config"], data_dir=file_name, num_classes=10
            )
        elif data_folder_name == "mnist":
            training_dataset = load_mnist(config=kwargs["config"], data_dir=file_name)
        else:
            raise NotImplementedError(
                "Invalid dataset ('Cifar100' or 'Cifar10' or 'MNIST')"
            )

        train_data = training_dataset["train_data"]
        train_labels = training_dataset["train_labels"]
        val_data = training_dataset["val_data"]
        val_labels = training_dataset["val_labels"]
        test_data = training_dataset["test_data"]
        test_labels = training_dataset["test_labels"]
        train_transforms = training_dataset["train_transforms"]
        val_transforms = training_dataset["val_transforms"]
        test_transforms = training_dataset["test_transforms"]
        interpretable_labels = training_dataset["interpretable_labels"]

        if data_folder_name == "cifar100" or data_folder_name == "cifar10":
            # For training
            train_dataset = CifarDataset(
                data=train_data,
                labels=train_labels,
                transform=train_transforms,
                interpretable_labels=interpretable_labels,
            )

            # For validation
            val_dataset = CifarDataset(
                data=val_data, labels=val_labels, transform=val_transforms
            )

        elif data_folder_name == "mnist":
            train_dataset = MNISTDataset(
                data=train_data, labels=train_labels, transform=train_transforms
            )

            # For validation
            val_dataset = MNISTDataset(
                data=val_data, labels=val_labels, transform=val_transforms
            )
            # For testing
            test_dataset = MNISTDataset(
                data=test_data, labels=test_labels, transform=test_transforms
            )

        else:
            raise NotImplementedError("Not 'CIFAR', nor 'MNIST'")

        return train_dataset, val_dataset, test_dataset
