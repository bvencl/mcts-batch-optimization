import os
import ast
import torch
import random
import argparse
import numpy as np
import pickle
from types import SimpleNamespace
from configparser import ConfigParser


def get_args():
    parser = argparse.ArgumentParser(description='Configuration file for training')
    parser.add_argument('--config', '-c', type=str, default='config.ini', help='Configuration file')
    return parser.parse_args()


def read(path):
    file_conf = ConfigParser()

    if not path.endswith('.ini'):
        path += '.ini'
    
    try:
        file_conf.read(path, encoding="utf8")
    except:
        raise FileNotFoundError(f"File {path} not found in configs folder")

    conf_dict = {}
    for section_name in file_conf.sections():
        d = {}
        for key, val in file_conf.items(section_name):
            d[key] = ast.literal_eval(val)

        item = SimpleNamespace(**d)
        conf_dict[section_name] = item
    conf = SimpleNamespace(**conf_dict)

    return conf


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def set_seeds(seed):
    """
    Setting packages seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
