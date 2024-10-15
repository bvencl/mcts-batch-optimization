import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, ResNet50_Weights
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, resnet50

from factory.base_factory import BaseFactory


class ModelFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        model = kwargs["config"].trainer.model_type
        pretrained = kwargs["config"].trainer.transfer_learning
        dataset = kwargs["config"].data_loader.dataset

        if dataset == "mnist" or "cifar10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        else:
            raise NotImplementedError("Dataset can be 'mnist', 'cifar10', 'cifar100'")

        if model == "mobilenet_v3_large":
            my_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)

            # Modify the first convolution layer to adapt to low resolution datasets
            my_model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                      bias=False)  # Adjusting the output layer
            my_model.classifier[3] = nn.Linear(my_model.classifier[3].in_features, num_classes)
            
        elif model == "mobilenet_v3_small":
            my_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)

            # Modify the first convolution layer to adapt to low resolution datasets
            my_model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                      bias=False)

            # Adjusting the output layer
            my_model.classifier[3] = nn.Linear(my_model.classifier[3].in_features, num_classes)

        elif model == "resnet50":
            my_model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

            # Modify the first convolution layer to adapt to low resolution datasets
            my_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # Modify the initial max pooling layer to prevent downsizing too fast
            my_model.maxpool = nn.Identity()

            # Adjusting the output layer
            my_model.fc = nn.Linear(my_model.fc.in_features, num_classes)

        else:
            raise NotImplementedError(
                "Valid options: mobilenet_v3_large, mobilenet_v3_small, resnet50")

        if kwargs["config"].trainer.dropout_off:
            cls.disable_dropout(my_model)
            print("Dropout is off")

        my_model.to(device)

        return my_model

    @staticmethod
    def disable_dropout(model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0