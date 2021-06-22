"""
A collection of utilities for searching image/ videos files.
"""
import os
import glob

VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi"]
IMAGE_EXTENSIONS = [".jpg", ".png"]


def is_image(path):
    """Checks if a path is an image

    Args:
        path: file path
    Returns:
        is_image (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_video(path):
    """Checks if a path is a video

    Args:
        path: file path

    Returns:
        is_video (bool): True or False
    """
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def grab_images_and_videos(adir):
    """Grabs image and video files

    Args:
        adir: directory with images

    Returns:
        files: image and video files
    """
    return grab_images(adir) + grab_videos(adir)


def grab_images(adir):
    """Grabs image files

    Args:
        adir: directory with images

    Returns:
        files: image files
    """
    return grab_files(adir, [".jpg", ".png"])


def grab_videos(adir):
    """Grabs videos in a directory

    Args:
        adir: directory or list of directory

    Returns:
        files: files with image/ video extension
    """
    files = grab_files(adir, VIDEO_EXTENSIONS)
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
    """Grabs files with allowed extensions

    Args:
        adir: directory
        extensions: allowed extensions

    Returns:
        files
    """
    all_files = []
    for ext in extensions:
        all_files += glob.glob(adir + "/*" + ext)
        all_files += glob.glob(adir + "/*" + ext.upper())
    return all_files
