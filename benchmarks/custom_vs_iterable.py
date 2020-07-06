import glob
import torch
import numpy as np
import threading
import cv2
import time
import random

from pytorch_streamloader.utils import grab_images_and_videos, normalize
from pytorch_streamloader.video_stream_dataset import make_video_dataset
from pytorch_iterable import MyIterableDataset, MultiStreamDataLoader

def time_iterator(dataloader, num_iter=100, main_thread_dt=0.0):
    print(dataloader)
    timings = []
    start = time.time()
    start2 = time.time()
    for idx, batch in enumerate(dataloader):
        timings.append(time.time()-start2)
        time.sleep(main_thread_dt)
        if idx >= num_iter:
            break
        start2 = time.time()
        if idx == 0:
            start = start2
    end = time.time()
    print(np.array(timings[1:]).mean())
    return end-start 

video_dir = "/home/etienneperot/workspace/data/slow-motion/video01/"
data_list = glob.glob(video_dir + "*.MP4")

loader_v1 = make_video_dataset(
        data_list,
        niter=100,
        tbins=10,
        max_frames=-1,
        random_start=False,
        num_workers=2,
        batchsize=4,
        height=240,
        width=320
    )
print('custom solution: ', time_iterator(loader_v1), ' s')


datasets = MyIterableDataset.split_datasets(
    data_list, batch_size=4, tbins=10, max_workers=2
)
loader_v2 = MultiStreamDataLoader(datasets)
print('pytorch iterable: ', time_iterator(loader_v2), ' s')

