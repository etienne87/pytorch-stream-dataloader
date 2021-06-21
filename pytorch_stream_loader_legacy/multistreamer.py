"""
MultiStreamer: creates a pool of N workers with their own queue.
each pool creates a "StreamGroup" that is user-defined,
it can holds several objects that streams their data to each queue.
the main process collects synchronously from each worker queue the last item.

The main usage of this is to stream temporally coherent batches to the same accelerator.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import threading

import numpy as np
import multiprocessing as python_mp

import torch
import torch.multiprocessing as mp
from torch._six import queue, container_abcs, string_classes
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

from collections import defaultdict

import time


class NumpySharedArray(object):
    """ This objects represent one shared array accross several workers
        It can be re-used for each conceptual large tensor we deliver
        to main process.
    Args:
        array_dim: dimensions
        num_videos_per_thread: number of videos per worker
        dtype: type of array
    """

    def __init__(self, array_dim, max_q_size, num_videos_per_thread, dtype):
        self.array_dim = array_dim
        self.num_videos_per_thread = num_videos_per_thread
        self.max_q_size = max_q_size
        array_dim2 = (self.max_q_size, self.num_videos_per_thread, *array_dim)

        mp_dtype = "f" if dtype == np.float32 else "b"
        self.m = mp.Array(mp_dtype, int(np.prod(array_dim2)), lock=mp.Lock())
        self.n = np.frombuffer(self.m.get_obj(), dtype=dtype).reshape(array_dim2)

    def acquire(self):
        self.m.acquire()

    def release(self):
        self.m.release()


class MultiStreamer(object):
    """
    Multithreaded Streaming for Temporally Coherent Batches
    Expects the "data" in tensor form with array_dim shape per thread.

    make_env: callback to user's group streaming batches
    array_dims: dic name to tensor shape
    max_q_size: fifo size per thread
    num_workers: number of threads
    collate_fn: user's collating data from multiple threads
    epoch: start epoch can be passed if data needs this information for epoch specific conditions of your stream_group.
    dtype: array dtype
    pin_memory: additional thread queue to "pin" data in host memory for cuda
    """

    def __init__(
        self,
        make_env,
        array_dims,
        batchsize,
        max_q_size,
        num_workers,
        collate_fn,
        epoch=0,
        dtype=np.uint8,
        main_thread=True,
        pin_memory=True,
    ):
        num_workers = min(max(num_workers, 1), batchsize)
        print("(MultiStreamer): num_workers:", num_workers)
        # queue per process
        self.worker_queues = [mp.Queue(maxsize=max_q_size) for i in range(num_workers)]

        # self.array_dim = array_dim
        self.num_workers = num_workers
        self.num_videos_per_worker = batchsize // num_workers

        print("num videos / worker: ", self.num_videos_per_worker)
        self.max_q_size = max_q_size
        self.batchsize = batchsize
        self.make_env = make_env

        if not isinstance(array_dims, dict):
            array_dims = {"data": array_dims}

        self.array_dims = array_dims
        self.batches = {}
        for k, array_dim in array_dims.items():
            self.batches[k] = np.zeros(
                (self.num_workers, self.num_videos_per_worker, *array_dim), dtype=dtype
            )

        self.shared_arrays = []
        for i in range(num_workers):
            dic = {}
            for k, array_dim in array_dims.items():
                dic[k] = NumpySharedArray(
                    array_dim, max_q_size, self.num_videos_per_worker, dtype
                )
            self.shared_arrays.append(dic)

        self.dataset = make_env(worker_id=0, num_workers=0, num_streams=0)
        self.max_iter = self.dataset.num_batches
        self.collate_fn = collate_fn
        self.epoch = epoch
        self.pin_memory = pin_memory
        self.main_thread = main_thread or pin_memory

    def __len__(self):
        """direct call to user argument max_iter"""
        return self.max_iter

    def worker_loop(self, i, m):
        """
        calls user's make_env function that is creating a stream group.
        the stream group streams temporally coherent batch portions needs to be ensured by user!

        stream_group class must contain a "next" function that collects from M streams,
        sometimes restarting them.

        """
        group = self.make_env(
            worker_id=i,
            num_workers=self.num_workers,
            num_streams=self.num_videos_per_worker,
            epoch=self.epoch,
        )
        j = 0
        for _ in range(group.num_batches):
            #  print('worker: ', i, ' wants to lock: ', j)
            [v.acquire() for v in m.values()]
            # print('worker: ', i, ' has locked: ', j)
            np_arrays = {k: v.n[j] for k, v in m.items()}
            info = group(np_arrays)
            self.worker_queues[i].put((j, info))
            j = (j + 1) % self.max_q_size

    def get_batch(self):
        """
        This collate a batch from all workers queues.
        It ensures all samples are collected synchronously.
        (This is why we do not push into a single queue)
        """
        batch = defaultdict(list)
        for t in range(self.num_workers):
            j, infos = self.worker_queues[t].get()
            for k, v in self.batches.items():
                self.batches[k][t] = self.shared_arrays[t][k].n[j]

            for k, v in infos.items():
                batch[k] += v

            for k, v in self.batches.items():
                self.shared_arrays[t][k].release()
            # print('main thread has released from worker queue: ', t, ' item : ', j)

        for k, v in self.batches.items():
            batch[k] = v.reshape(self.batchsize, *self.array_dims[k])

        out = self.collate_fn(batch)
        return out

    def join_streams_thread(self, out_queue, device_id, done_event):
        """
        additional thread putting data into a queue to be collected from __iter__
        """
        torch.set_num_threads(1)
        torch.cuda.set_device(device_id)

        while not done_event.is_set():
            data = self.get_batch()
            if (
                self.pin_memory
                and not done_event.is_set()
                and not isinstance(data, ExceptionWrapper)
            ):
                data = pin_memory(data)
            out_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)

    def __iter__(self):
        """
        main process: setup workers, setup joining thread, collect max_iter batches from main data queue.
        """
        procs = [
            mp.Process(target=self.worker_loop, args=(i, m), daemon=True)
            for i, m in enumerate(self.shared_arrays)
        ]

        [p.start() for p in procs]

        if self.main_thread:
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

        for i in range(self.max_iter):
            data = (
                self._data_queue.get(timeout=100000)
                if self.main_thread
                else self.get_batch()
            )
            yield data

        if self.pin_memory:
            self._join_memory_thread_done_event.set()
        [p.terminate() for p in procs]
        self.epoch += 1
