from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, glob, random, time
import numpy as np
import cv2
import torch
from utils import grab_images_and_videos
from functools import partial

from data.opencvstreamer import OpenCVStream
from data.ffmpegstreamer import FFMpegStream

# from data.multistreamer_iq import MultiStreamer

from data.multistreamer import MultiStreamer
from data.stream_dataset import StreamDataset
from data.camstreamer import CameraImageStream
from data.data_augmentation import DataAugmentedStream


# Data Augmented backend
def da_opencv_video_backend(*args, **kwargs):
    return DataAugmentedStream(OpenCVStream, *args, **kwargs)


def opencv_video_backend(*args, **kwargs):
    return OpenCVStream(*args, **kwargs)


# refactor of StreamGroup
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
        self.video_backend = kwargs.get("video_backend", da_opencv_video_backend)
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


class StreamGroup(object):
    def __init__(
        self,
        proc_id=0,
        num_procs=1,
        num_envs=3,
        epoch=0,
        niter=100,
        max_frames=1000,
        utbins=1,
        **kwargs
    ):
        filenames = kwargs.get("filenames", [])
        self.random_start = kwargs.get("random_start", True)
        self.video_backend = kwargs.get("video_backend", opencv_video_backend)

        num_procs = max(1, num_procs)

        num_files = len(filenames) // num_procs
        start = proc_id * num_files
        end = (proc_id + 1) * num_files

        print(
            "(StreamGroup) proc: ",
            proc_id,
            "/",
            num_procs,
            " num files: ",
            len(filenames),
            "start: ",
            start,
            " end: ",
            end,
            " max_frames=",
            max_frames,
            " random_start: ",
            self.random_start,
        )
        self.filenames = filenames[start:end]  # split into several threads

        random.shuffle(self.filenames)
        self.height, self.width = kwargs.get("height", 480), kwargs.get("width", 640)
        self.file_iter = 0
        self.num_envs = num_envs
        self.max_iter = niter
        self.num_tbins = utbins
        self.max_frames = max_frames
        self.reset()
        self.iter = 0

    def reset(self):
        self.streams = []
        self.file_iter = 0
        random.shuffle(self.filenames)
        for i in range(self.num_envs):
            filename = self.filenames[self.file_iter]
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
                        self.filenames[self.file_iter],
                        self.height,
                        self.width,
                        max_frames=self.max_frames,
                        random_start=self.random_start,
                    )
                ]
            self.file_iter = (self.file_iter + 1) % len(self.filenames)

    def reload(self, env_idx):
        print(self.filenames[self.file_iter])
        self.streams[env_idx].reload(self.filenames[self.file_iter])
        self.file_iter = (self.file_iter + 1) % len(self.filenames)

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
                    self.reload(i)
                    ret, frame = self.streams[i].run()
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


def make_video_stream(
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


def show_stream(
    *all_dirs,
    niter=100,
    max_frames=10,
    tbins=10,
    utbins=1,
    batchsize=4,
    num_workers=1,
    height=360,
    width=640,
    random_start=False,
    main_thread_dt=0.4,
    viz=True
):
    from torchvision.utils import make_grid
    from utils import normalize

    all_filenames = []
    for adir in all_dirs:
        if os.path.isdir(adir):
            all_filenames += grab_images_and_videos(adir)

    color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    dataloader = make_video_stream(
        all_filenames,
        niter=niter,
        tbins=tbins,
        max_frames=max_frames * tbins,
        utbins=utbins,
        random_start=random_start,
        num_workers=num_workers,
        batchsize=batchsize,
        height=height,
        width=width,
    )
    start = 0
    t0 = 0
    nrows = 2 ** ((batchsize.bit_length() - 1) // 2)
    for epoch in range(10):
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataloader):

            # x = data["data"].cuda()  # we do put the stuff on the GPU
            x = data["data"]

            # torch.cuda.synchronize()
            total_time = time.time() - t0
            rate = total_time / (1 + batch_idx)
            print("rate: ", rate)
            print(time.time() - start, " s ", batch_idx, "/", len(dataloader))
            if viz:
                tbins, batchsize = data["data"].shape[:2]
                for t in range(len(data["data"])):
                    grid = (
                        make_grid(data["data"][t], nrow=nrows)
                        .permute(1, 2, 0)
                        .numpy()
                        .copy()
                    )
                    grid = cv2.putText(
                        grid, str(t), (10, len(grid) - 30), font, 1.0, color, 2
                    )
                    cv2.imshow("batch", grid.astype(np.uint8))
                    key = cv2.waitKey(5)
                    if key == 27:
                        return
            else:
                time.sleep(main_thread_dt)
            if batch_idx == 0:
                t0 = time.time()
            start = time.time()


if __name__ == "__main__":
    import fire

    fire.Fire()
