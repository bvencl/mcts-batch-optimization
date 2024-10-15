import sys

sys.setrecursionlimit(1000000)

from factory import *
from utils.utils import *
from utils.trainer import Trainer


def main():
    args = get_args()
    config = read(args.config)
    config.data_loader.custom_sampler = 0

    seed = config.trainer.seed
    set_seeds(seed)

    train_dataset, val_dataset, test_dataset = DatasetFactory.create(config=config)
    train_loader, val_loader, sampler = DataLoaderFactory.create(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    my_model = ModelFactory.create(config=config)
    my_criterion, my_optimizer, lr_scheduler = AgentFactory.create(
        config=config, model=my_model
    )
    my_callbacks = CallbackFactory.create(
        config=config,
        model=my_model,
        val_loader=val_loader,
        criterion=my_criterion,
    )

    trainer = Trainer(
        config=config,
        criterion=my_criterion,
        optimizer=my_optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=my_callbacks,
        model=my_model,
    )

    trainer.train()


if __name__ == "__main__":
    main()
