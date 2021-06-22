"""
The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.
"""
import random
import time
import torch
import numpy as np

from pytorch_stream_dataloader.stream_dataset import StreamDataset
from itertools import chain, cycle
from collections import deque
from torch.utils.data import IterableDataset, DataLoader
from pytorch_stream_dataloader.utils import split_batch_size, split_dataset_sizes



class StreamDataLoader(object):
    """StreamDataLoader
    Wraps around the DataLoader to handle the asynchronous batches.

    Args:
        dataset (StreamDataset): dataset streaming multiple iterables
        num_workers (int): number of workers
        collate_fn (function): function to collate batch parts
    """

    def __init__(self, dataset, num_workers, collate_fn):
        self.dataset = dataset
        num_workers = min(dataset.batch_size, num_workers)
        assert isinstance(dataset, StreamDataset)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            drop_last=False)
        self.collate_fn = collate_fn
        self.num_workers = max(1, num_workers)

    def __iter__(self):
        cache = [deque([]) for i in range(self.num_workers)]
        for data in self.dataloader:
            data, worker_id = data
            cache[worker_id].append(data)
            num = sum([len(v) > 0 for v in cache])
            if num == self.num_workers:
                batch = [item.popleft() for item in cache]
                batch = chain.from_iterable(iter(batch))
                batch = self.collate_fn(batch)
                yield batch
        # Empty remaining cache
        if self.dataset.padding_mode == 'zeros':
            split_sizes = split_batch_size(self.dataset.batch_size, self.num_workers)
            num_actives = self.num_workers
            while num_actives > 0:
                values = []
                for i, deq in enumerate(cache):
                    if not len(deq):
                        value = [self.dataset.fill_value] * split_sizes[i]
                    else:
                        value = deq.popleft()
                    values.append(value)
                batch = chain.from_iterable(values)
                yield self.collate_fn(batch)
                num_actives = sum([len(deq) for deq in cache])
