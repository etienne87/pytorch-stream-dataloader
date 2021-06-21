"""
Module that enables Parallel Multistreaming.

We define an IterableDataset that streams several iterables.

When fed to a Pytorch DataLoader with batch_size=None,
this streams batches from one worker at a time.
This has the effect of enabling parallel streaming.

The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Notice that you can also avoid using this fusion and just use
a regular DataLoader, and have multiple neural networks indexed
by worker's id.

Note of implementation:


Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.

"""
import random
import time
import torch
import numpy as np

from torch.utils.data import IterableDataset, DataLoader
from itertools import chain, cycle
from collections import deque


def split_batch_size(batch_size, num_workers):
    """Returns a list of batch_size

    Args:
        batch_size (int): total batch size
        num_workers (int): number of workers
    """
    num_workers = min(num_workers, batch_size)
    split_size = batch_size // num_workers
    total_size = 0
    split_sizes = [split_size] * (num_workers - 1)
    split_sizes += [batch_size - sum(split_sizes)]
    return split_sizes


def split_dataset_sizes(stream_list, split_sizes):
    """Splits with different sizes

    Args:
        stream_list (list): list of stream path
        split_sizes (list): batch size per worker
    """
    out = []
    start = 0
    total = sum(split_sizes)
    for split_size in split_sizes[:-1]:
        num = int(split_size / total * len(stream_list))
        end = start + num
        out.append(stream_list[start:end])
        start = end
    out.append(stream_list[start:])
    return out


def resample_to_batch_size(stream_list, batch_size):
    """Resamples list to fit batch_size iterators

    Args:
        stream_list (list): list of streams
        batch_size (int): batch size
    """
    stream_list = random.sample(stream_list, len(stream_list)) +\
        random.choices(stream_list, k=batch_size - len(stream_list) % batch_size)
    return stream_list

class StreamDataset(IterableDataset):
    """Stream Dataset
    An Iterable Dataset zipping a group of iterables streams together.

    Args:
        stream_list (list): list of streams (path/ metadata)
        streamer (object): an iterator (user defined)
        batch_size (int): total batch size
        padding_mode (str): "zeros" "data" or "none", see "get_zip" function
        fill_value (object): padding value
    """

    def __init__(self, stream_list, streamer, batch_size, padding_mode, fill_value):
        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        assert padding_mode in ['zeros', 'data']
        if padding_mode == 'zeros':
            assert fill_value is not None

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = int(time.time()) + worker_id
        np.random.seed(seed)
        random.seed(seed)

    def _worker_init_fn(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        num_workers = 1 if worker is None else worker.num_workers
        split_sizes = split_batch_size(self.batch_size, num_workers)
        stream_groups = split_dataset_sizes(self.stream_list, split_sizes)
        split_size = split_sizes[worker_id]
        stream_group = stream_groups[worker_id]
        random.shuffle(stream_group)
        return split_size, stream_group

    def __iter__(self):
        """Iterates over stream files

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        self._set_seed()

        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        split_size, stream_list = self._worker_init_fn()
        if len(stream_list) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            raise StopIteration

        """
        Just-in-time mapping
        The scheduling is done as we iterate.
        """
        iterators = [iter(self.streamer(stream_list[i])) for i in range(split_size)]
        actives = [1 for i in range(len(iterators))]
        num_actives = sum(actives)
        file_pos = split_size-1
        while num_actives:
            values = []
            for i, it in enumerate(iterators):
                try:
                    value = next(it)
                except StopIteration:
                    file_pos += 1
                    actives[i] = 1 * (file_pos < len(stream_list))
                    if self.padding_mode == 'data' or actives[i]:
                        num = file_pos % len(stream_list)
                        iterators[i] = iter(self.streamer(stream_list[num]))
                        value = next(iterators[i])
                    elif self.padding_mode == 'zeros':
                        value = self.fill_value

                values.append(value)
            yield tuple(values), worker_id
            num_actives = sum(actives)


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
