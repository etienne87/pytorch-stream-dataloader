from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, glob, random, time
import numpy as np
import cv2


def moving_average(item, alpha):
    return (1 - alpha) * item + alpha * np.random.randn(*item.shape)


def computeC2MC1(R_0to1, tvec_0to1, R_0to2, tvec_0to2):
    R_1to2 = R_0to2.dot(R_0to1.T)
    tvec_1to2 = R_0to2.dot(-R_0to1.T.dot(tvec_0to1)) + tvec_0to2
    return R_1to2, tvec_1to2


def generate_homography(rvec1, tvec1, rvec2, tvec2, nt, K, Kinv, d):
    R_0to1 = cv2.Rodrigues(rvec1)[0].transpose()
    tvec_0to1 = np.dot(-R_0to1, tvec1.reshape(3, 1))

    R_0to2 = cv2.Rodrigues(rvec2)[0].transpose()
    tvec_0to2 = np.dot(-R_0to2, tvec2.reshape(3, 1))

    # view 0to2
    nt1 = R_0to1.dot(nt.T).reshape(1, 3)
    H_0to2 = R_0to2 - np.dot(tvec_0to2.reshape(3, 1), nt1) / d
    G_0to2 = np.dot(K, np.dot(H_0to2, Kinv))

    return G_0to2


class Camera(object):
    def __init__(self, height, width):
        self.K = np.array(
            [[width / 2, 0, width / 2], [0, width / 2, height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        self.Kinv = np.linalg.inv(self.K)

        self.rvec1 = np.array([0, 0, 0], dtype=np.float32)
        self.tvec1 = np.array([0, 0, 0], dtype=np.float32)
        self.nt = np.array([0, 0, -1], dtype=np.float32).reshape(1, 3)

        self.rvec_amp = np.random.rand(3) * 0.25
        self.tvec_amp = np.random.rand(3) * 0.5

        self.rvec_speed = np.random.choice([5e-3, 1e-2, 1e-3])
        self.tvec_speed = np.random.choice([5e-3, 1e-2, 1e-3])

        self.rvec_amp[2] = 0.0

        self.tshift = np.random.randn(3)
        self.rshift = np.random.randn(3)
        self.d = 1
        self.time = 0

    def __call__(self):
        self.tshift = moving_average(self.tshift, 1e-4)
        self.rshift = moving_average(self.rshift, 1e-4)
        rvec2 = self.rvec_amp * np.sin(self.time * self.rvec_speed + self.rshift)
        tvec2 = self.tvec_amp * np.sin(self.time * self.tvec_speed + self.tshift)
        G_0to2 = generate_homography(
            self.rvec1, self.tvec1, rvec2, tvec2, self.nt, self.K, self.Kinv, self.d
        )
        G_0to2 /= G_0to2[2, 2]
        self.time += 1
        return G_0to2


class CameraImageStream(object):
    def __init__(self, image_filename, height, width, max_frames=30):
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.reload(image_filename)
        self.border_mode = cv2.BORDER_CONSTANT

    def pos_frame(self):
        return self.iter

    def reload(self, image_filename):
        self.filename = image_filename
        frame = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        self.frame = cv2.resize(frame, (self.width, self.height), 0, 0, cv2.INTER_AREA)
        self.cam = Camera(self.height, self.width)
        self.iter = 0

    def __len__(self):
        return self.max_frames

    def run(self):
        if self.max_frames > -1 and self.iter >= self.max_frames:
            return False, None

        G_0to2 = self.cam()
        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.width, self.height),
            borderMode=self.border_mode,
        )
        self.iter += 1
        return True, out


def just_open_one_stream(image_filename, max_frames=1000):
    video = CameraMotionImageStream(image_filename, 360, 640, max_frames=max_frames)
    for i in range(max_frames):
        frame = video.run()
        cv2.imshow("frame", frame)
        cv2.waitKey(5)


if __name__ == "__main__":
    import fire

    fire.Fire()
