import os
import numpy as np
from uuid import uuid4


class Node:
    def __init__(
        self,
        epoch,
        core_acc=None,
        core_loss=None,
        parent=None,
        batch_idx=None,
        available_batch_idxs=None,
    ):
        self._visits = 0
        self._value = 0.0
        self._core_acc = core_acc
        self._core_loss = core_loss
        self.parent = parent
        self.children = []
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.available_batch_idxs = available_batch_idxs
        self.version = str(uuid4())

        self._init_path(str(epoch), self.version)

    @property
    def core_acc(self):
        return self._core_acc  # 1g

    @core_acc.setter
    def core_acc(self, value):
        self._core_acc = value

    @property
    def core_loss(self):
        return self._core_loss  # 1g

    @core_loss.setter
    def core_loss(self, value):
        self._core_loss = value

    @property
    def n(self):
        return self._visits

    @property
    def q(self):
        return self._value

    def _init_path(self, epoch, version):
        root_folder = "nodes"
        path = os.path.join(root_folder, epoch, version)
        os.makedirs(path, exist_ok=True)
        self.path = os.path.join(path, "temp_model.pth")

    def backpropagate(self, score, score_multiplier):
        self._visits += 1
        self._value += score * score_multiplier
        if self.parent:
            self.parent.backpropagate(score, score_multiplier)

    def best_child(self, c_value=1.41):
        exploit = []
        explore = []

        for child in self.children:
            try:
                exploit.append(child.q / child.n)
                explore.append(c_value * np.sqrt(2 * np.log(self.n) / child.n))
            except ZeroDivisionError:
                exploit.append(0)
                explore.append(np.inf)

        choices_weights = [i + j for i, j in zip(exploit, explore)]
        max_value = np.max(choices_weights)
        indices = np.where(choices_weights == max_value)[0]

        return self.children[np.random.choice(indices)]

    def print_best_child_by_mean_accuracy(self):
        mean_score = []

        for child in self.children:
            mean_score.append(child.q / child.n)  # exploit tagokat egy listába rakja

        if mean_score:  # ha nem üres
            return self.children[
                np.argmax(mean_score)
            ].print_best_child_by_mean_accuracy()
        else:
            best_acc = self.core_acc
            print("Best mean accuracy: {:.4} in layer {}".format(best_acc * 100, self.epoch))

    def print_best_child_by_val_accuracy(self):
        max = []

        for child in self.children:
            max.append(child.q)  # val_acc egy listába rakja

        if max:  # ha nem üres
            return self.children[
                np.argmax(max)
            ].print_best_child_by_val_accuracy()
        else:
            best_acc = self.core_acc
            print("Best validation accuracy: {:.4} in layer {}".format(best_acc * 100, self.epoch))
