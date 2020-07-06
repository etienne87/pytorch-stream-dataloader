import glob
import torch
import numpy as np
import threading
import cv2
import time
import random

from torchvision.utils import make_grid
from torch.utils.data import IterableDataset, DataLoader

from torch._six import queue, container_abcs, string_classes
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

from itertools import chain, cycle, islice
from pytorch_streamloader.video_streamer import OpenCVStream


class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size=4, tbins=5):
        self.data_list = data_list
        self.batch_size = batch_size
        self.tbins = tbins

    @property
    def shuffle_data_list(self):
        return random.sample(self.data_list, len(self.data_list))

    def process_data(self, data):
        stream = OpenCVStream(data, height=240, width=320, max_frames=-1)
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1

        out = []
        for frame_num, x in enumerate(stream):
            ret, x = x
            if x is None:
                break
            out.append(x[None])
            if len(out) == self.tbins:
                y = np.concatenate(out)
                out = []
                yield y

    def get_stream(self, data_list):
        tmp = map(self.process_data, iter(data_list))
        out = chain.from_iterable(tmp)
        return out

    def __iter__(self):
        return zip(
            *[self.get_stream(self.shuffle_data_list) for _ in range(self.batch_size)]
        )

    @classmethod
    def split_datasets(cls, data_list, batch_size, tbins, max_workers):
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        # Here is an example which utilises a single worker to build the entire batch.
        split_size = batch_size // num_workers
        return [
            cls(data_list, batch_size=split_size, tbins=tbins)
            for _ in range(num_workers)
        ]


class MultiStreamDataLoader:
    def __init__(self, datasets, pin_memory=True):
        self.datasets = datasets
        self.pin_memory = pin_memory

    def get_stream_loaders(self):
        dataloaders = [
            DataLoader(dataset, num_workers=1, batch_size=None, pin_memory=True)
            for dataset in self.datasets
        ]
        return zip(*dataloaders)

    def join_streams_thread(self, out_queue, device_id, done_event):
        """
        additional thread putting data into a queue to be collected from __iter__
        """
        torch.set_num_threads(1)
        torch.cuda.set_device(device_id)

        for idx, batch_parts in enumerate(self.get_stream_loaders()):
            data = list(chain(*batch_parts))

            data = torch.cat([item[:, None] for item in data], dim=1)
            if (
                not done_event.is_set()
                and not isinstance(data, ExceptionWrapper)
            ):
                data = pin_memory(data)

            out_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)

        self._join_memory_thread_done_event.set()

    def __iter__(self):
        # define a thread for collation & memory pinning here
        if self.pin_memory:
            self._join_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()
            self.join_memory_thread = threading.Thread(
                target=self.join_streams_thread,
                args=(
                    self._data_queue,
                    torch.cuda.current_device(),
                    self._join_memory_thread_done_event,
                ),
            )
            self.join_memory_thread.daemon = True
            self.join_memory_thread.start()
           
            while not self._join_memory_thread_done_event.is_set():
                batch = self._data_queue.get(timeout=100000)
                batch = {'data':batch}
                yield batch
            self.join_memory_thread.join()
        else:
            # Single-Process
            for batch_parts in self.get_stream_loaders():
                data = list(chain(*batch_parts))
                batch = torch.cat([item[:, None] for item in data], dim=1)
                batch = {'data':batch}
                yield batch
