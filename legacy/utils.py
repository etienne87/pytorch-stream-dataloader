"""
utils
"""
import os
import glob
import time
import torch
import numpy as np


def grab_images_and_videos(adir):
    return grab_images(adir) + grab_videos(adir)


def grab_images(adir):
    return grab_files(adir, [".jpg", ".png"])


def grab_videos(adir):
    files = grab_files(adir, [".MP4", ".mp4", ".mov", ".MOV", ".m4v", ".avi"])
    # detect jpg directories
    others = os.listdir(adir)
    for sdir in others:
        subdir = os.path.join(adir, sdir)
        imgs = grab_images(subdir)
        if len(imgs):
            # detect number of digits
            img_name = os.path.basename(imgs[0])
            ext = os.path.splitext(img_name)[1]
            nd = sum(c.isdigit() for c in img_name)
            redir = os.path.join(subdir, "%" + str(nd) + "d" + ext)
            files.append(redir)

    return files


def grab_files(adir, extensions):
    all_files = []
    for ext in extensions:
        all_files += glob.glob(adir + "/*" + ext)
    return all_files


def filter_outliers(input_val, num_std=3):
    val_range = num_std * input_val.std()
    img_min = input_val.mean() - val_range
    A
    img_max = input_val.mean() + val_range
    if isinstance(input_val, torch.Tensor):
        normed = torch.min(torch.max(input_val, img_min), img_max)
    else:
        normed = np.clip(input_val, img_min, img_max)  # clamp
    return normed


def normalize(im):
    return (im - im.min()) / (im.max() - im.min() + 1e-5)


def cuda_tick():
    torch.cuda.synchronize()
    return time.time()


def cuda_time(func):
    def wrapper(*args, **kwargs):
        start = cuda_tick()
        out = func(*args, **kwargs)
        end = cuda_tick()
        rt = end - start
        freq = 1.0 / rt
        if freq > 0:
            print(freq, " it/s @ ", func)
        else:
            print(rt, " s/it @ ", func)
        return out

    return wrapper
