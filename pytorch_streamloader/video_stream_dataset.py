from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, glob, random, time
import numpy as np
import cv2
import torch
from functools import partial

from pytorch_streamloader.multistreamer import MultiStreamer
from pytorch_streamloader.stream_dataset import StreamDataset

from pytorch_streamloader.video_streamer import OpenCVStream
from pytorch_streamloader.virtual_camera_streamer import CameraImageStream


def opencv_video_backend(*args, **kwargs):
    return OpenCVStream(*args, **kwargs)


class VideoStreams(StreamDataset):
    def __init__(
        self,
        stream_files=[],
        worker_id=0,
        num_workers=1,
        num_streams=3,
        num_batches=100,
        num_tbins=1,
        epoch=0,
        **kwargs
    ):
        self.height, self.width = kwargs.get("height", 480), kwargs.get("width", 640)
        self.random_start = kwargs.get("random_start", True)
        self.video_backend = kwargs.get("video_backend", opencv_video_backend)
        self.max_frames = 1000
        self.epoch = epoch
        super(VideoStreams, self).__init__(
            stream_files, worker_id, num_workers, num_streams, num_batches, num_tbins
        )

    def reset_streams(self):
        self.streams = []
        self.file_iter = 0
        random.shuffle(self.stream_files)
        print("reset: ", self.num_streams)
        for i in range(self.num_streams):
            assert len(self.stream_files) > 0
            filename = self.stream_files[self.file_iter]
            extension = os.path.splitext(filename)[1]
            if extension in [".jpg", ".png"] and not "%" in filename:
                self.streams += [
                    CameraImageStream(
                        self.filenames[self.file_iter],
                        self.height,
                        self.width,
                        max_frames=self.max_frames,
                    )
                ]
            else:
                self.streams += [
                    self.video_backend(
                        self.stream_files[self.file_iter],
                        self.height,
                        self.width,
                        max_frames=self.max_frames,
                        random_start=self.random_start,
                    )
                ]
            self.file_iter = (self.file_iter + 1) % len(self.stream_files)

    def reload_stream(self, idx):
        self.streams[idx].reload(self.stream_files[self.file_iter])
        self.file_iter = (self.file_iter + 1) % len(self.stream_files)

    def __call__(self, arrays_dic):
        batchsize, tbins = arrays_dic["data"].shape[:2]
        assert len(self.streams) == batchsize
        mask = np.zeros((batchsize, tbins), dtype=np.uint8)
        filenames = []
        times = []
        self.iter += 1
        for i, stream in enumerate(self.streams):
            filenames_i = []
            times_i = []
            for t in range(tbins):
                ret, frame = next(stream)
                if not ret and t % self.num_tbins != 0:
                    continue
                while not ret:
                    self.reload_stream(i)
                    ret, frame = next(self.streams[i])
                    if not ret:
                        print(
                            "(StreamGroup) Error while loading: ",
                            self.streams[i].filename,
                        )
                mask[i, t] = self.streams[i].iter > 1
                arrays_dic["data"][i, t] = frame[..., None]
        return {"resets": [mask], "iter": [self.iter - 1]}



def collate_fn(data):
    batch, resets = data["data"], data["resets"]
    batch = torch.from_numpy(batch).permute(1, 0, 4, 2, 3).contiguous()
    resets = (
        torch.cat([torch.from_numpy(item) for item in resets]).permute(1, 0).float()
    )
    return {"data": batch, "resets": resets}


def make_video_dataset(
    all_filenames,
    niter=100,
    tbins=10,
    utbins=1,
    num_workers=1,
    batchsize=8,
    max_frames=100,
    start_epoch=0,
    height=480,
    width=640,
    random_start=True,
    streamer_func=opencv_video_backend,
):
    array_dims = (tbins, height, width, 1)
    make_env = partial(
        VideoStreams,
        stream_files=all_filenames,
        num_batches=niter,
        height=height,
        width=width,
        max_frames=max_frames,
        num_tbins=utbins,
        random_start=random_start,
        video_backend=streamer_func,
    )
    dataset = MultiStreamer(
        make_env,
        array_dims,
        batchsize=batchsize,
        max_q_size=4,
        num_workers=num_workers,
        collate_fn=collate_fn,
        epoch=start_epoch,
        dtype=np.uint8,
        main_thread=0,
        pin_memory=0,
    )

    return dataset

