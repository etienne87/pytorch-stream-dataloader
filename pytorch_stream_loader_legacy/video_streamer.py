from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, glob, random, time
import numpy as np
import cv2
import torch


class OpenCVStream(object):
    def __init__(
        self,
        video_filename,
        height,
        width,
        seek_frame=0,
        max_frames=-1,
        random_start=True,
        rgb=False,
    ):
        self.height = height
        self.width = width
        self.random_start = random_start
        self.max_frames = max_frames
        self.rgb = rgb
        self.reload(video_filename, seek_frame)

    def original_size(self):
        height, width = (
            self.cap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH),
        )
        return int(height), int(width)

    def reload(self, video_filename, seek_frame=-1):
        self.filename = video_filename
        self.cap = cv2.VideoCapture(video_filename)
        self.iter = 0
        if self.random_start and seek_frame == -1:
            num_frames = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
            seek_frame = random.randint(0, num_frames // 2)
            if seek_frame > 0:
                self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, seek_frame)
        else:
            seek_frame = 0 if seek_frame == -1 else seek_frame
            if seek_frame > 0:
                self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, seek_frame)
        self.start = seek_frame
        if self.height == -1 or self.width == -1:
            self.height, self.width = self.original_size()

    def pos_frame(self):
        return self.start + self.iter

    def __len__(self):
        if self.max_frames > -1:
            return self.max_frames
        else:
            num_frames = int(self.cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
            return num_frames - self.start

    def __next__(self):
        if not self.cap:
            return False, None

        if not self.cap or (self.max_frames > -1 and self.iter >= self.max_frames):
            return False, None

        ret, frame = self.cap.read()
        if ret:
            if not self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), 0, 0, cv2.INTER_AREA)
            self.iter += 1
        return ret, frame

    def __iter__(self):
        return self
