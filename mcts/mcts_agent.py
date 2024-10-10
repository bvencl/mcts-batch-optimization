import copy
import os
import sys
sys.path.append('/')
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mcts.node import Node
from utils.visualise import *
from utils.validate import validate_model

class MCTS:
    def __init__(self, config):
        self.config = config
        self.branching_factor = self.config.mcts.branching_factor
        self.n_iters = self.config.mcts.n_iters
        self.c_param = self.config.mcts.c_param
        self.checkpoint = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        if self.config.mcts.branching_mode == 'max':
            self.is_fully_expanded = self.is_fully_expanded_max_branching
        elif self.config.mcts.branching_mode == 'factor':
            self.is_fully_expanded = self.is_fully_expanded_branching_factor

    def search(self, model, val_loader, criterion, optimizer, sampler, model_checkpoint=None, neptune_namespace=None, visualise=False):
        self._init_search(model, sampler, model_checkpoint, val_loader, criterion)

        for i in range(self.n_iters):
            print("Iteration #{0}".format(i))

            v = self.tree_policy(criterion, optimizer, sampler)

            val_loss , val_acc = self.rollout(v, sampler, val_loader, criterion, optimizer)
            
            if self.is_terminal_node(v):
                self.checkpoint(val_loss, val_acc, self._base_model, neptune_namespace)
            
            if visualise:
                self.best_branch_visualisation(self.root)

            v.backpropagate(val_acc)

            self.root.print_best_child_by_val_accuracy()
            save_tree(self.root, 'all')

        return self.root

    def _init_search(self, model, sampler, model_checkpoint, val_loader, criterion):
        self._base_model = model
        self.checkpoint = model_checkpoint if model_checkpoint is not None else None
        
        if self.config.paths.load_model:
            self._base_model.load_state_dict(torch.load(self.config.paths.load_model, weights_only=True)) 
                     
        elif os.path.exists(self.config.paths.model_checkpoint_path + "checkpoint.pth"):
            self._base_model.load_state_dict(torch.load(self.config.paths.model_checkpoint_path + "checkpoint.pth", weights_only=True))
            val_loss, val_acc = validate_model(model=self._base_model, data_loader=val_loader, criterion=criterion)
            print("_init_search:\nVal loss: {:.4f}, Val accuracy: {:.2f}%".format(val_loss, val_acc * 100.))
            
        self.root = Node(0, available_batch_idxs=np.arange(sampler.num_batches))
        torch.save(self._base_model.state_dict(), self.root.path)

    def tree_policy(self, criterion, optimizer, sampler):
        current_node = self.root
        
        while not self.is_terminal_node(current_node):
            if not self.is_fully_expanded(current_node, sampler):
                return self.expand(current_node, sampler, criterion, optimizer)
            else:
                current_node = current_node.best_child(self.c_param)
                
        return current_node

    @staticmethod
    def is_terminal_node(node: Node):
        return True if len(node.available_batch_idxs) == 0 else False

    def is_fully_expanded_branching_factor(self, node: Node, *args):
        return True if len(node.children) == self.branching_factor else False

    @staticmethod
    def is_fully_expanded_max_branching(node: Node, sampler):
        return True if len(node.children) == sampler.num_batches else False

    def expand(self, node, sampler, criterion, optimizer):
        self._base_model.load_state_dict(torch.load(node.path, weights_only=True))
        batch_idx = np.random.choice(node.available_batch_idxs)

        self.train_model(batch_idx, sampler, criterion, optimizer)

        available_batch_idxs = node.available_batch_idxs[node.available_batch_idxs != batch_idx]
        child_node = Node(node.epoch + 1, parent=node, batch_idx=batch_idx, available_batch_idxs=available_batch_idxs)
        node.children.append(child_node)
        torch.save(self._base_model.state_dict(), child_node.path)

        return child_node

    def train_model(self, batch_idx, sampler, criterion, optimizer):
        self._base_model.train()

        inputs, labels, _ = sampler.get_batch(batch_idx)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        optimizer.zero_grad()
        outputs = self._base_model(inputs)
        loss = criterion(outputs, labels.clone())

        loss.backward()
        optimizer.step()

    def rollout(self, node, sampler, val_loader, criterion, optimizer):
        self._base_model.load_state_dict(torch.load(node.path, weights_only=True))
        batch_sequence = copy.deepcopy(node.available_batch_idxs)
        np.random.shuffle(batch_sequence) 

        if self.config.mcts.rollout:
            for batch_idx in batch_sequence:
                self.train_model(batch_idx, sampler, criterion, optimizer)

        val_loss, val_acc = validate_model(model=self._base_model, data_loader=val_loader, criterion=criterion)
        node.core_loss, node.core_acc = val_loss, val_acc

        return val_loss, val_acc

    def best_branch(self, node):
        best = deepcopy(node)
        batch_idxs = []

        def _best_branch(node):
            if len(node.children) == 0:
                return

            best_child = node.best_child(self.c_param)
            node.children = [best_child]
            batch_idxs.append(best_child.batch_idx)

            return _best_branch(best_child)

        _best_branch(best)

        return best, batch_idxs

    def best_branch_visualisation(self, node):

        best = self.best_branch(node)

        start = False

        if len(best.children) > 0:
            self.best_branch_visualisation(best.children[0])
        else:
            start = True

        rewards = []
        start = best

        while len(start.children) > 0:
            rewards.append(start.core_reward)
            start = start.children[0]

        plt.plot(rewards)
        plt.ylim(0, 200)
        plt.xlim(0, self.train_episodes)
        plt.pause(0.0001)
        plt.cla()
        plt.show(block=False)