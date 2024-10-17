import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger(NeptuneLogger):
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        parameters = {
            "number_of_epochs": self._config.trainer.n_epochs,
            "dropout_off": self._config.trainer.dropout_off,
            "train_data_size": self._config.data_loader.num_data_samples_train,
            "mcts_data_size": self._config.data_loader.num_data_samples_mcts,
            "datasize": (
                self._config.data_loader.num_data_samples_train
                if not self._config.data_loader.custom_sampler
                else self._config.data_loader.num_data_samples_mcts
            ),
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
            "lr_start": self._config.agent.lr_start,
            "lr_warmup_end": self._config.agent.lr_warmup_end,
            "lr_end": self._config.agent.lr_end,
            "lr_decay_type": self._config.agent.lr_decay_type,
            "exp_gamma": self._config.agent.exp_gamma,
            "warmup_epochs": self._config.agent.warmup_epochs,
            "loss": self._config.agent.loss,
            "optimizer": self._config.agent.optimizer,
            "branching_mode": self._config.mcts.branching_mode,
            "branching_factor": self._config.mcts.branching_factor,
            "c_param": self._config.mcts.c_param,
            "rollout": self._config.mcts.rollout,
            "mcts_iterations_start": self._config.mcts.mcts_iterations_start,
            "mcts_iterations_max": self._config.mcts.mcts_iterations_max,
            "default_exploit_multiplier": self._config.mcts.default_exploit_multiplier,
            "increasing_exploit_multiplier": self._config.mcts.increasing_exploit_multiplier,
            "max_exploit_multiplier": self._config.mcts.max_exploit_multiplier,
            "min_exploit_multiplier": self._config.mcts.min_exploit_multiplier,
            "exploit_multiplier_kick_in": self._config.mcts.exploit_multiplier_kick_in,
            "exploit_multiplier_steps": self._config.mcts.exploit_multiplier_steps,
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
