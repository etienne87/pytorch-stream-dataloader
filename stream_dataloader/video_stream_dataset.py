"""
We use Decord to read videos.
You can choose between using decord videoloader directly or customizing the Iterator
"""
import decord
import numpy as np
import torch
from .stream_dataloader import StreamDataLoader, StreamDataset
from .files import grab_images_and_videos


class VideoStream(object):
    def __init__(self, path, start_frame, end_frame, height, width, num_tbins, time_last=True):
        self.__dict__.update(**locals())
        self.reader = decord.VideoReader(self.path, ctx=decord.cpu(0))
        self.orig_shape = self.reader.get_batch([0]).shape[1:3]
        self.end_frame = len(self.reader) if self.end_frame == -1 else self.end_frame

    def get_orig_size(self):
        return self.orig_shape

    def __iter__(self):
        self.reader.seek(self.start_frame)
        num = self.end_frame-self.start_frame
        for i in range(0, num, self.num_tbins):
            end = min(len(self.reader)-1, i+self.num_tbins)
            if end-i <= 0:
                raise StopIteration
            frames = self.reader.get_batch([j for j in range(i,end)]).asnumpy()
            frames = torch.from_numpy(frames) #t,h,w,c
            if self.time_last:
                frames = frames.permute(1,2,3,0)
            yield frames, 0


def pad_collate_fn(data_list, time_last=True):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    images, _ = zip(*data_list)
    time_dim = -1 if time_last else 0
    max_len = max([item.shape[time_dim] for item in images])
    b = len(images)
    if time_last:
        h, w, c = images[0].shape[:3]
    else:
        h, w, c = images[0].shape[1:3]
    shape = (b,h,w,c,max_len) if time_last else (max_len,b,h,w,c)
    out_images = torch.zeros(shape, dtype=images[0].dtype)
    for i in range(b):
        video = images[i]
        ilen = video.shape[-1]
        if time_last:
            out_images[i, ..., :ilen] = video
            out_images[i, ..., ilen:] = video[..., ilen-1:]
        else:
            out_images[:ilen,i] = video
            out_images[ilen:,i] = video[ilen-1]

    return {'images': out_images}


class VideoLoader(StreamDataLoader):
    def __init__(self, path, batch_size, num_workers):
        files = grab_images_and_videos(path)
        def iterator_fun(file_path):
            return VideoStream(file_path, start_frame=0, end_frame=-1, height=0, width=0, num_tbins=10)
        dataset = StreamDataset(files, iterator_fun, batch_size, "data", None)
        super().__init__(dataset, num_workers, pad_collate_fn)


