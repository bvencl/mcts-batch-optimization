from data_loader.load_cifar_100 import load_cifar_100
from data_loader.load_cifar_10 import load_cifar_10


def load_cifar(config, data_dir, num_classes=100):
    if num_classes == 100:
        cifar_data = load_cifar_100(config, data_dir)
    elif num_classes == 10:
        cifar_data = load_cifar_10(config, data_dir)
    else:
        raise ValueError

    return cifar_data
