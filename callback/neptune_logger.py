import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger(NeptuneLogger):
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        parameters = {
            "train_data_size": self._config.data_loader.num_data_samples_train,
            "mcts_data_size": self._config.data_loader.num_data_samples_mcts,
            "val_and_test_data_size": self._config.data_loader.num_data_samples_val_test,
            "val_test_spplit": self._config.data_loader.val_test_split,
            "batch_size_train": self._config.data_loader.batch_size_train,
            "batch_size_validate": self._config.data_loader.batch_size_validate,
            "batch_size_test": self._config.data_loader.batch_size_test,
            "dataset": self._config.data_loader.dataset,
            "data_augmentation": self._config.data_loader.augmentation,
            "model_checkpoint": self._config.callbacks.model_checkpoint,
            "model_checkpoint_type": self._config.callbacks.model_checkpoint_type,
            "num_workers": self._config.data_loader.num_workers,
            "seed": self._config.trainer.seed,
            "model_type": self._config.trainer.model_type,
            "transfer_learning": self._config.trainer.transfer_learning,
            "lr_decay": self._config.agent.lr_decay,
            "starting_learning_rate": self._config.agent.starting_learning_rate,
            "lr_min": self._config.agent.lr_min,
            "lr_decay_type": self._config.agent.lr_decay_type,
            "warmup_epochs": self._config.agent.warmup_epochs,
            "warmup_lr_high": self._config.agent.warmup_lr_high,
            "loss": self._config.agent.loss,
            "optimizer": self._config.agent.optimizer,
        }
        parameters = {k: (v if v is not None else 0) for k, v in parameters.items()}
        self.run["parameters"] = parameters

        self.logger = NeptuneLogger(run=self.run, model=model)

    def save_model(self, path):
        self.run["models/model"].upload(path)

    def stop(self):
        self.run.stop()

    def __del__(self):
        if self._params_hook_handler is not None:
            self._params_hook_handler.remove()
