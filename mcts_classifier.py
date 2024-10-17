import sys
import warnings
sys.setrecursionlimit(1000000)
from mcts.mcts_agent import MCTS
from factory import *
from utils.utils import *
from utils.trainer import Trainer

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

def main():
    args = get_args()
    config = read(args.config)
    config.data_loader.custom_sampler = 1

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
    mcts = MCTS(config=config)

    trainer = Trainer(
        config=config,
        criterion=my_criterion,
        optimizer=my_optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=my_callbacks,
        model=my_model,
        sampler=sampler,
        mcts=mcts,
    )

    trainer.train()


if __name__ == "__main__":
    main()
