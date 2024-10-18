import neptune
from neptune_pytorch import NeptuneLogger


class CustomNeptuneLogger(NeptuneLogger):
    def __init__(self, token, project, config):
        self.run = neptune.init_run(api_token=token, project=project)
        self._config = config
        self.logger = None

    def start_logging(self, model):
        
        parameters = {}
        
        def add_param(key, value):
            parameters[key] = value
                
        add_param("number_of_epochs", self._config.trainer.n_epochs)
        add_param("dropout_off", self._config.trainer.dropout_off)
        add_param("datasize", self._config.data_loader.num_data_samples_train if not self._config.data_loader.custom_sampler else self._config.data_loader.num_data_samples_mcts)
        add_param("val_and_test_data_size", self._config.data_loader.num_data_samples_val_test)
        add_param("batch_size_train", self._config.data_loader.batch_size_train)
        add_param("batch_size_validate", self._config.data_loader.batch_size_validate)
        add_param("dataset", self._config.data_loader.dataset)
        add_param("data_augmentation", self._config.data_loader.augmentation)
        add_param("model_checkpoint", self._config.callbacks.model_checkpoint)
        add_param("model_checkpoint_type", self._config.callbacks.model_checkpoint_type)
        add_param("num_workers", self._config.data_loader.num_workers)
        add_param("seed", self._config.trainer.seed)
        add_param("model_type", self._config.trainer.model_type)
        add_param("transfer_learning", self._config.trainer.transfer_learning)
        add_param("loss", self._config.agent.loss)
        add_param("optimizer", self._config.agent.optimizer)
        
        if self._config.data_loader.batch_size_test:
            add_param("batch_size_test", self._config.data_loader.batch_size_test)
            add_param("val_test_split", self._config.data_loader.val_test_split)
            
        
        if self._config.agent.lr_decay:
            add_param("lr_decay_type", self._config.agent.lr_decay_type)
            add_param("lr_start", self._config.agent.lr_start)
            if self._config.agent.lr_decay_type == "warmup_cos":
                add_param("lr_warmup_end", self._config.agent.lr_warmup_end)
                add_param("lr_end", self._config.agent.lr_end)
                add_param("warmup_epochs", self._config.agent.warmup_epochs)
            if self._config.agent.lr_decay_type == "exp":
                add_param("exp_gamma", self._config.agent.exp_gamma)
            if self._config.agent.lr_decay_type == "cos":
                add_param("lr_end", self._config.agent.lr_end)                
        
        if self._config.data_loader.custom_sampler:
            add_param("branching_mode", self._config.mcts.branching_mode)
            add_param("branching_factor", self._config.mcts.branching_factor)
            add_param("c_param", self._config.mcts.c_param)
            add_param("rollout", self._config.mcts.rollout)
            add_param("mcts_iterations_start", self._config.mcts.n_iters_start)
            add_param("mcts_iterations_max", self._config.mcts.n_iters_max)
            add_param("default_exploit_multiplier", self._config.mcts.default_exploit_multiplier)
            add_param("increasing_exploit_multiplier", self._config.mcts.increasing_exploit_multiplier)
            add_param("max_exploit_multiplier", self._config.mcts.max_exploit_multiplier)
            add_param("min_exploit_multiplier", self._config.mcts.min_exploit_multiplier)
            add_param("exploit_multiplier_kick_in", self._config.mcts.exploit_multiplier_kick_in)
            add_param("exploit_multiplier_steps", self._config.mcts.exploit_multiplier_steps)
        
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
