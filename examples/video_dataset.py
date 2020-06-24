from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import time
import numpy as np
import cv2

from torchvision.utils import make_grid
from pytorch_streamloader.utils import grab_images_and_videos, normalize
from pytorch_streamloader.video_stream_dataset import make_video_dataset

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

    all_filenames = []
    for adir in all_dirs:
        if os.path.isdir(adir):
            all_filenames += grab_images_and_videos(adir)

    color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    dataloader = make_video_dataset(
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
    fire.Fire(show_stream)
