import os
import torch
from callback.neptune_logger import CustomNeptuneLogger
from callback.model_checkpoint import ModelCheckpoint
from utils.validate import validate_model


class CallbackFactory:

    @classmethod
    def create(cls, **kwargs):
        my_callbacks = {}
        config = kwargs["config"]

        if config.callbacks.neptune_logger:
            token = config.callbacks.neptune_token
            project = config.callbacks.neptune_project
            if token is None or project is None:
                raise ValueError("Neptune token adn/or project are/is not defined.")

            neptune_logger_callback = CustomNeptuneLogger(
                token=token, project=project, config=config
            )
            neptune_logger_callback.start_logging(kwargs["model"])
            my_callbacks["neptune_logger"] = neptune_logger_callback

        if config.callbacks.model_checkpoint:
            path = config.paths.model_checkpoint_path
            loaded_checkpoint = os.path.exists(path + "checkpoint.pth")
            if loaded_checkpoint and config.callbacks.remove_previous_checkpoint_at_start:
                os.remove(path + "checkpoint.pth")
                loaded_checkpoint = False
            elif loaded_checkpoint:
                kwargs["model"].load_state_dict(torch.load(path + "checkpoint.pth"), weights_only=True)
                val_loss, val_acc = validate_model(
                    kwargs["model"], kwargs["val_loader"], kwargs["criterion"]
                )
            verbose = config.callbacks.model_checkpoint_verbose

            checkpoint = ModelCheckpoint(
                type=config.callbacks.model_checkpoint_type,
                verbose=verbose,
                path=path,
                neptune_logger=(
                    neptune_logger_callback if config.callbacks.neptune_logger else None
                ),
                loaded_values=(val_loss, val_acc) if loaded_checkpoint else None,
            )
            my_callbacks["model_checkpoint"] = checkpoint
            if verbose and loaded_checkpoint:
                print(
                    f"Model loaded to model checkpoint with {val_acc} validation accuracy and {val_loss} validation loss!"
                )

        return my_callbacks
