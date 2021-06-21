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



class TextStream(object):
    def __init__(self, text, tbins):
        self.text = np.fromstring(text, dtype=np.uint8)
        self.iter = 0
        self.tbins = tbins

    def __len__(self):
        return 100

    def __next__(self):
        if self.iter >= len(self.text):
            return None
        frame = self.text[self.iter:self.iter+self.tbins] 
        self.iter += self.tbins
        return frame

    def reload(self, text):
        self.text = text


class TextStreams(StreamDataset):
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
        self.max_frames = 1000
        self.epoch = epoch
        self.stream_files = stream_files
        super(TextStreams, self).__init__(
            stream_files, worker_id, num_workers, num_streams, num_batches, num_tbins
        )

    def reset_streams(self):
        self.streams = []
        for i in range(self.num_streams):
            self.streams.append(TextStream(self.stream_files[self.stream_iter], self.num_tbins))
            self.stream_iter += 1

    def reload_stream(self, idx):
        self.streams[idx].reload(self.words[self.stream_iter])
        self.stream_iter = (self.stream_iter + 1) % len(self.stream_files)

    def __call__(self, arrays_dic):
        batchsize, tbins = arrays_dic["data"].shape[:2]
        assert len(self.streams) == batchsize
        mask = np.zeros((batchsize), dtype='u8')
        filenames = []
        times = []
        self.iter += 1
        for i, stream in enumerate(self.streams):
            filenames_i = []
            times_i = []
            frame = next(stream)
            while frame is None:
                self.reload_stream(i)
                frame = next(self.streams[i])
            mask[i] = self.streams[i].iter > 1
            arrays_dic["data"][i, :len(frame)] = frame[:,None]
        return {"resets": [mask]}


def collate_fn(data):
    batch = data["data"]
    return batch


def make_text_dataset(
    words,
    num_iter=10,
    num_tbins=80,
    num_workers=1,
    batchsize=8,
    max_frames=100,
    start_epoch=0,
):
    array_dims = (num_tbins, 1)
    make_env = partial(
        TextStreams,
        stream_files=words,
        num_batches=num_iter,
        max_frames=max_frames,
        num_tbins=num_tbins,
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



if __name__ == '__main__':
    import time
    import requests
    from bs4 import BeautifulSoup    
    from GoogleNews import GoogleNews
    googlenews = GoogleNews()
    googlenews = GoogleNews(lang='en')
    googlenews = GoogleNews(period='d')
    googlenews = GoogleNews(start='02/01/2020',end='02/28/2020')
    googlenews.setlang('en')
    googlenews.setperiod('d')
    googlenews.setTimeRange('02/01/2020','02/28/2020')
    googlenews.search('APPL')
    googlenews.getpage(2)
    x = googlenews.result()
    for item in x:
        web_link = item['link']
        
        start = time.time()
        page_source = requests.get(web_link)
        soup = BeautifulSoup(page_source.text, "lxml")
        print('s: ', time.time()-start)
        try: 
            text = soup.find('article').text
            #print(text)
        except:
            continue


    #text = googlenews.gettext()
    #print(text[0])
    # import pdb;pdb.set_trace()

