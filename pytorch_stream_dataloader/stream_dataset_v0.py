"""
Module that enables Parallel Multistreaming.

We define an IterableDataset that streams several iterables.

When fed to a Pytorch DataLoader with batch_size=None,
this streams batches from one worker at a time.
This has the effect of enabling parallel streaming.
"""
import random
import time
import torch
import numpy as np

from torch.utils.data import IterableDataset, DataLoader
from pytorch_stream_dataloader.utils import split_batch_size, split_dataset_sizes
from pytorch_stream_dataloader.join_data_thread import JoinDataThread



class StreamDataset(IterableDataset):
    """Stream Dataset
    An Iterable Dataset zipping a group of iterables streams together.

    Args:
        stream_list (list): list of streams (path/ metadata)
        streamer (object): an iterator (user defined)
        batch_size (int): total batch size
        padding_mode (str): "zeros" "data" or "none", see "get_zip" function
        padding_value (object): padding value
    """
    def __init__(self, stream_list, streamer, batch_size, padding_mode, padding_value, pos, num_actives, mutex):
        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        assert padding_mode in ['zeros', 'data']
        if padding_mode == 'zeros':
            assert padding_value is not None
        self.pos = pos
        self.mutex = mutex
        self.num_actives = num_actives

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = int(time.time()) + worker_id
        np.random.seed(seed)
        random.seed(seed)

    def init_position(self):
        self.mutex.acquire()
        self.pos.value = 0
        self.num_actives.value = 0
        self.mutex.release()

    def __iter__(self):
        """Iterates over stream files

        Note: Here we use a mutex (WIP, pytest not working!)

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        self._set_seed()

        #initialization this should be done in worker_init_fn
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0

        num_workers = 1 if worker is None else worker.num_workers
        split_sizes = split_batch_size(self.batch_size, num_workers)
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        split_size = split_sizes[worker_id]

        if len(self.stream_list) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            raise StopIteration

        """
        Just-in-time mapping
        The scheduling is done as we iterate.

        EDIT 9/7/2021: The position in the stream is shared accross workers
        This allows us to avoid the non ideal pre-iteration splitting of the dataset
        """
        def increment_pos():
            self.mutex.acquire()
            pos = self.pos.value
            stream = self.stream_list[pos%len(self.stream_list)]
            new_pos = pos + 1
            self.pos.value = new_pos
            self.mutex.release()
            return stream

        iterators = []
        for i in range(split_size):
            stream = increment_pos()
            stream = iter(self.streamer(stream))
            iterators.append(stream)

        actives = [1 for i in range(len(iterators))]
        _num_actives = sum(actives)
        self.mutex.acquire()
        self.num_actives.value += _num_actives
        self.mutex.release()

        while self.num_actives.value:
            values = []
            for i, it in enumerate(iterators):
                try:
                    value = next(it)
                    assert value is not None
                except StopIteration:
                    if actives[i] and (self.pos.value >= len(self.stream_list)):
                        self.mutex.acquire()
                        self.num_actives.value -= 1
                        self.mutex.release()
                    actives[i] = 1 * (self.pos.value < len(self.stream_list))
                    if self.padding_mode == 'data' or actives[i]:
                        assert stream is not None, self.pos.value
                        stream = increment_pos()
                        iterators[i] = iter(self.streamer(stream))
                        value = next(iterators[i])
                    elif self.padding_mode == 'zeros':
                        value = self.padding_value

                values.append(value)
            yield tuple(values), worker_id

