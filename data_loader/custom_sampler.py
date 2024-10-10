import numpy as np
from torch.utils.data import Sampler
import torch
import math


class FixedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.data_source_dims = self.data_source.data[0].shape
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.batches = self._create_batches()

    def _create_batches(self):
        # Create the list of batches
        batches = []
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = list(range(start_idx, end_idx))
            batches.append(batch_indices)
        return batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def get_batch(self, batch_idx):
        if batch_idx < 0 or batch_idx >= self.num_batches:
            raise IndexError("Batch index out of range")

        indices = self.batches[batch_idx]
        inputs = torch.empty((len(indices), 3, self.data_source_dims[-2], self.data_source_dims[-1]))
        labels = torch.empty((len(indices)), dtype=torch.long)
        idxs = torch.empty((len(indices)))

        for i in range(len(indices)):
            input, label, idx = self.data_source[indices[i]]
            inputs[i], labels[i], idxs[i] = input, label, idx
        
        return inputs, labels, idxs
