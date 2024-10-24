import torch
import torch.optim as optim

from utils.agent_utils import *
from factory.base_factory import BaseFactory
from utils.agent_utils import FocalLoss


class AgentFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        config = kwargs["config"]
        model = kwargs["model"]
        loss = config.agent.loss
        optimizer_name = config.agent.optimizer

        if loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif loss == "focal_loss":
            criterion = FocalLoss()
        else:
            raise NotImplementedError("Invalid loss type ('cross_entropy' or 'focal_loss')")

        if optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config.agent.lr_start)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.agent.lr_start)
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=config.getfloat("agent", "lr_start"))
        else:
            raise NotImplementedError("Invalid optimizer type ('sgd' or 'adam' or 'rmsprop')")

        if config.agent.lr_decay:
            decay_strategy = config.agent.lr_decay_type

            if decay_strategy == "cos":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.trainer.n_epochs, verbose=config.agent.lr_verbose)
            elif decay_strategy == "warmup_cos":
                T_max = config.trainer.n_epochs - config.agent.warmup_epochs
                lr_scheduler = WarmupCosineAnnealingLR(optimizer=optimizer,
                                                       T_max=T_max,
                                                       warmup_epochs=config.agent.warmup_epochs,
                                                       eta_min=config.agent.lr_end, # végén ide tart
                                                       eta_max=config.agent.lr_warmup_end,# ide megy fel warmup végén
                                                       verbose=config.agent.lr_verbose) 
            elif decay_strategy == "lin":
                lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, verbose=config.agent.lr_verbose)
            elif decay_strategy == "exp":
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.agent.exp_gamma, verbose=config.agent.lr_verbose)
            else:
                raise ValueError("Invalid learning rate scheduler "
                                 "('cos' or 'warmup_cos' or 'lin' or 'exp')")
        else:
            lr_scheduler = None

        return criterion, optimizer, lr_scheduler