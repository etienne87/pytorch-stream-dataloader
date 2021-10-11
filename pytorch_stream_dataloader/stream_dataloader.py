"""
The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.
"""
import time
import torch
import numpy as np
import multiprocessing

from pytorch_stream_dataloader.stream_dataset import StreamDataset
from itertools import chain, cycle
from collections import deque
from torch.utils.data import IterableDataset, DataLoader
from pytorch_stream_dataloader.utils import split_batch_size, split_dataset_sizes


import random


class StreamDataLoader(object):
    """StreamDataLoader

    Wraps around the DataLoader to handle the asynchronous batches.
    We now handle one single list of streams read from multiple workers with a mutex.

    Args:
        iterator_fun (lambda): function to create one stream
        batch_size (int): number of streams read at the same time
        num_workers (int): number of workers
        collate_fn (function): function to collate batch parts
        padding_mode (str): "data" or "zeros", what to do when all streams have been read but you still but one thread of streaming needs to output something
        padded_value (object): object or None
    """
    def __init__(self, files, iterator_fun, batch_size, num_workers, collate_fn, padding_mode, padding_value=None):
        mutex = multiprocessing.Lock()
        pos = multiprocessing.Value('i', 0)
        num_actives = multiprocessing.Value('i', 0)
        dataset = StreamDataset(files, iterator_fun, batch_size, padding_mode, padding_value, pos, num_actives, mutex)
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
        self.dataloader.dataset.shuffle()
        self.dataloader.dataset.init_position()

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
        # Assert no value is a true value
        for fifo in cache:
            if not len(cache):
                continue
            while len(fifo):
                item = fifo.pop()[0]
                if item != self.dataset.padding_value:
                    assert 0, 'code is broken, cache contained real data'
