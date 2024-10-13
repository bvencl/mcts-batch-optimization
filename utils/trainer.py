import torch
import pickle
import copy
import shutil
import os
from utils.utils import set_seeds
from utils.visualise import save_tree
from utils.validate import validate_model


class Trainer:
    def __init__(self, config, criterion, optimizer, lr_scheduler, train_loader, val_loader, callbacks, model,
                 sampler=None, mcts=None):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.model = model
        self.model_path = config.paths.model_path
        self.model_name = config.paths.model_name
        self.neptune_namespace = None
        self.checkpoint = None
        
        self.sampler = sampler
        self.mcts = mcts
        self.train = self.train_classic if self.sampler is None else self.train_mcts

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.n_epochs = self.config.trainer.n_epochs
        self.neptune_logger = True if self.config.callbacks.neptune_logger else False
        if self.neptune_logger:
            neptune = self.callbacks["neptune_logger"]
            self.neptune_namespace = neptune.run[neptune.logger.base_namespace]
            
        self.model_checkpoint = True if self.config.callbacks.model_checkpoint else False
        if self.model_checkpoint:
            self.checkpoint = self.callbacks["model_checkpoint"]

    def train_classic(self, batch_sequence_idxs=None):
        if self.neptune_logger:
            self.neptune_namespace["metrics/val_acc"].append(0.0)
            self.neptune_namespace["metrics/val_loss"].append(0.0)
        # Training & Validation
        train_loss, val_loss = [], []
        train_accuracy, val_accuracy = [], []

        for epoch in range(self.n_epochs) if batch_sequence_idxs is None else [0]:
            self.model.train()
            running_loss, correct_train = 0.0, 0
            train_loader_iter = iter(self.train_loader) if batch_sequence_idxs is None else None
            length = min(len(self.train_loader), len(batch_sequence_idxs)) if batch_sequence_idxs is not None else len(self.train_loader)
            for i in range(length):
                inputs, labels, idx = self.get_batch(batch_sequence_idxs, i, train_loader_iter)
                # print(idx)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.clone())

                if self.neptune_logger:
                    self.neptune_namespace["metrics/train_loss"].append(loss.item())

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(inputs),
                                                                               len(self.train_loader.dataset),
                                                                               100. * i/len(self.train_loader),
                                                                               loss.item()))

            train_loss.append(running_loss / len(self.train_loader))
            last_train_acc = 100. * correct_train / len(self.train_loader.dataset)
            train_accuracy.append(last_train_acc)

            if self.neptune_logger:
                self.neptune_namespace["metrics/train_acc"].append(copy.deepcopy(last_train_acc / 100.))

            self.model.eval()
            running_loss, correct_val = 0.0, 0

            with torch.no_grad():
                for inputs, labels, _ in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.clone())



                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_val += (predicted == labels).sum().item()

            val_loss.append(running_loss / len(self.val_loader))
            last_val_acc = 100.0 * correct_val / len(self.val_loader.dataset)
            val_accuracy.append(last_val_acc)
            
            if self.neptune_logger:
                self.neptune_namespace["metrics/val_loss"].append(copy.deepcopy(val_loss[-1]))
                self.neptune_namespace["metrics/val_acc"].append(copy.deepcopy(last_val_acc / 100.0))

            print(
                f"Epoch {epoch + 1}/{self.n_epochs} - Train loss: {train_loss[-1]:.4f},"
                f"Train accuracy: {train_accuracy[-1]:.2f}%, Val loss: {val_loss[-1]:.4f},"
                f"Val accuracy: {val_accuracy[-1]:.2f}%")

            if self.model_checkpoint:
                self.checkpoint(val_loss[-1], val_accuracy[-1]/100.0, self.model, self.neptune_namespace)

            if self.neptune_logger:
                self.neptune_namespace["metrics/lr"].append(float(self.lr_scheduler.get_lr()[-1]))
                self.lr_scheduler.step()


        history = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy}

        with open('training_history.pkl', 'wb') as file:
            pickle.dump(history, file)
        if not self.model_checkpoint:
            file_name = os.path.join(self.model_path, self.model_name + ".pth")
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            torch.save(self.model.state_dict(), file_name)
            if self.neptune_logger:
                self.callbacks["neptune_logger"].save_model(file_name)
                self.callbacks["neptune_logger"].stop()
            

    def train_mcts(self):
        if self.neptune_logger:
                    self.neptune_namespace["metrics/val_acc"].append(0.0)
                    self.neptune_namespace["metrics/val_loss"].append(0.0)
        try:
            for epoch in range(self.n_epochs):
                print(f'Setting learning rate to {self.lr_scheduler.get_lr()[0]}!')
                _ = self.mcts.search(self.model, self.val_loader, self.criterion, self.optimizer, self.sampler, self.checkpoint, neptune_namespace = self.neptune_namespace, visualise=False)
                # ! Itt nem feltétlenül a checkpointot kellene betölteni
                self.model.load_state_dict(torch.load(self.config.paths.model_checkpoint_path + "checkpoint.pth", weights_only=True))               
                val_loss, val_acc = validate_model(model=self.model, data_loader=self.val_loader, criterion=self.criterion)
                
                if self.neptune_logger:
                    self.neptune_namespace["metrics/val_acc"].append(copy.deepcopy(val_acc))
                    self.neptune_namespace["metrics/val_loss"].append(val_loss)
                
                if  self.neptune_logger:
                    self.neptune_namespace["metrics/lr"].append(self.lr_scheduler.get_lr()[0])
                if self.config.agent.lr_decay:
                    self.lr_scheduler.step()
                    
                
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} - Val loss: {val_loss:.4f},"
                    f"Val accuracy: {100 * val_acc:.2f}%")
                
                save_tree(self.mcts.root, 'all')
                shutil.rmtree('nodes/')
        
        except KeyboardInterrupt:
            print("Interrupted the training, shutting down...") 
        torch.save(self.model.state_dict(), self.model_path + self.model_name + ".pth")
        if self.neptune_logger:
            self.callbacks["neptune_logger"].save_model(self.model_path + self.model_name + ".pth")
            self.callbacks["neptune_logger"].stop()
            


    def get_batch(self, batch_sequence_idxs, i, train_loader_iter):
        if batch_sequence_idxs is not None:
            inputs, labels, _ = self.sampler.get_batch(batch_sequence_idxs[i])
            idx = batch_sequence_idxs[i]
        else:
            inputs, labels, _ = next(train_loader_iter)
            idx = None

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        return inputs, labels, idx
