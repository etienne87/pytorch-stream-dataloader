from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, glob, random, time
import numpy as np
import torch
from functools import partial
from data.multistreamer import MultiStreamer, StreamGroup


class IndexDataset(StreamGroup):
    def __init__(
        self,
        worker_id=0,
        num_workers=1,
        num_streams=2,
        epoch=0,
        niter=100,
        max_frames=1000,
        tbins=3,
        **kwargs
    ):
        super(IndexDataset, self).__init__(
            worker_id,
            num_workers,
            num_streams,
            epoch,
            niter,
            max_frames,
            tbins,
            **kwargs
        )

    def select_partition(self, **kwargs):
        indexes = [chr(ord("A") + i) for i in range(27)]
        num_files = len(indexes) // self.num_workers
        start = self.worker_id * num_files
        end = (self.worker_id + 1) * num_files
        self.indexes = indexes[start:end]
        self.frame_num = [0 for _ in self.indexes]

    def reset(self):
        pass

    def next(self, array_dict):
        for i, (idx, frame_num) in enumerate(zip(self.indexes, self.frame_num)):
            for t in range(self.tbins):
                array_dict["data"][i, t] = ord(idx) * 1000 + frame_num
                self.frame_num[i] += 1
        return {}


def collate_fn(data):
    batch = data["data"]
    batch = torch.cat(*batch)
    return batch


def test_simple_case(tbins=5, batchsize=4, niter=100, num_workers=2):
    array_dims = (tbins, 1)
    make_env = partial(IndexDataset, niter=niter, tbins=tbins)
    dataset = MultiStreamer(
        make_env,
        array_dims,
        batchsize=batchsize,
        max_q_size=5,
        num_workers=num_workers,
        collate_fn=collate_fn,
        epoch=0,
        dtype=np.float32,  # problem if uint32
    )

    for batch in dataset:
        print(batch)


if __name__ == "__main__":
    import fire

    fire.Fire(test_simple_case)
