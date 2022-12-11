"""
We use Decord to read videos.
You can choose between using decord videoloader directly or customizing the Iterator

Notes: it seems the fastest is to use decord with num_workers=0!
"""
import numpy as np
import torch

from pytorch_stream_dataloader.stream_dataloader import StreamDataLoader
from pytorch_stream_dataloader.stream_dataset import StreamDataset
from pytorch_stream_dataloader.utils import split_batch_size, split_dataset_sizes
from examples.files import grab_images_and_videos

import decord
import skvideo.io
import itertools
import time
import os
import pickle


class DummyVideoStream(object):
    def __init__(self, height, width, num_tbins):
        self.__dict__.update(**locals())
        self.num_frames = np.random.randint(1,10)*num_tbins

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        num = self.num_frames
        for i in range(0, num, self.num_tbins):
            end = min(num, i+self.num_tbins)
            if end-i <= 0:
                raise StopIteration
            frames = torch.randint(0,255,(self.num_tbins, 3, self.height,
                self.width), dtype=torch.uint8)
            yield frames


class DecordVideoStream(object):
    def __init__(self, path, start_frame, end_frame, height, width, num_tbins):
        self.__dict__.update(**locals())
        self.reader = decord.VideoReader(self.path, ctx=decord.cpu(0))
        self.orig_shape = self.reader.get_batch([0]).shape[1:3]
        self.end_frame = len(self.reader) if self.end_frame == -1 else self.end_frame

    def __len__(self):
        return len(self.reader)

    def get_orig_size(self):
        return self.orig_shape

    def __iter__(self):
        self.reader.seek(self.start_frame)
        num = self.end_frame-self.start_frame
        for i in range(0, num, self.num_tbins):
            end = min(len(self.reader)-1, i+self.num_tbins)
            if end-i <= 0:
                raise StopIteration
            frames = self.reader.get_batch([j for j in range(i, end)]).asnumpy()
            frames = torch.from_numpy(frames)  # t,h,w,c
            frames = frames.permute(0, 3, 1, 2)  # t,c,h,w
            yield frames


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class ScikitVideoStream(object):
    def __init__(self, path, start_frame, end_frame, height, width, num_tbins):
        self.__dict__.update(**locals())
        self.metadata = skvideo.io.ffprobe(self.path)
        self.orig_shape = int(self.metadata["video"]["@height"]), int(self.metadata["video"]["@width"])
        reader = skvideo.io.FFmpegReader(self.path)
        self.len = reader.getShape()[0]
        self.end_frame = reader.getShape()[0] if self.end_frame == -1 else self.end_frame

    def __len__(self):
        return self.len

    def get_reader(self):
        fps = eval(self.metadata["video"]["@avg_frame_rate"])
        start_ts = self.start_frame / fps
        input_dict = {'-ss': str(start_ts)}
        output_dict = {}
        reader = skvideo.io.vreader(self.path, num_frames=self.end_frame-self.start_frame,
                                    inputdict=input_dict, outputdict=output_dict)
        return reader

    def __iter__(self):
        fps = eval(self.metadata["video"]["@avg_frame_rate"])
        start_ts = self.start_frame / fps
        input_dict = {'-ss': str(start_ts)}
        output_dict = {}
        reader = skvideo.io.vreader(self.path, num_frames=self.end_frame-self.start_frame,
                                    inputdict=input_dict, outputdict=output_dict)
        for frames in grouper(reader, self.num_tbins):
            frames = [frame for frame in frames if frame is not None]
            frames = [frame[None] for frame in frames]
            frames = np.concatenate(frames)  # t,h,w,c
            frames = torch.from_numpy(frames)
            frames = frames.permute(0, 3, 1, 2)  # t,c,h,w
            yield frames


def pad_collate_fn(data_list):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    images = data_list
    max_len = max([item.shape[0] for item in images])
    b = len(images)
    _, c, h, w = images[0].shape
    shape = (max_len, b, c, h, w)
    out_images = torch.zeros(shape, dtype=images[0].dtype)
    for i in range(b):
        video = images[i]
        ilen = video.shape[0]
        out_images[:ilen, i] = video
        out_images[ilen:, i] = video[ilen-1]

    return out_images


def cut_videos(files, max_frames):
    """
    cut files into multiple duration with start_time/ end_time
    """
    print('this may take a while...')
    cuts = []
    for file_path in files:
        #ds = decord.VideoReader(file_path, ctx=decord.cpu(0))
        ds = skvideo.io.FFmpegReader(file_path).getShape()
        num = len(ds)
        cuts += [(file_path, i, i+max_frames) for i in range(0, num, max_frames)]
    return cuts


def cut_videos_load_rewrite(path, max_frames):
    """
    handle caching
    """
    cut_list = os.path.join(path, 'cut_video.pkl')
    do_write = True
    if os.path.exists(cut_list):
        files_pkl, max_frames_pkl = pickle.load(open(cut_list, 'rb'))
        if max_frames_pkl == max_frames:
            return files_pkl
    files = grab_images_and_videos(path)
    files = cut_videos(files, max_frames)
    pickle.dump((files, max_frames), open(cut_list, 'wb'))
    return files


class VideoLoader(StreamDataLoader):
    def __init__(self, path, batch_size, num_workers, max_frames=500, backend='scikit'):
        if path:
            files = cut_videos_load_rewrite(path, max_frames)
        else:
            files = [(None,None,None) for i in range(100)] 
            backend = 'dummy'

        def iterator_fun(args):
            file_path, start_frame, end_frame = args
            if backend == 'scikit':
                return ScikitVideoStream(file_path, start_frame, end_frame, height=0, width=0, num_tbins=10)
            elif backend == 'decord':
                return DecordVideoStream(file_path, start_frame, end_frame, height=0, width=0, num_tbins=10)
            elif backend == 'dummy':
                return DummyVideoStream(height=240, width=320, num_tbins=10)
            else:
                raise Exception("backend is not supported")

        padded_value = None
        super().__init__(files, iterator_fun, batch_size, num_workers, pad_collate_fn, "data", padded_value)
