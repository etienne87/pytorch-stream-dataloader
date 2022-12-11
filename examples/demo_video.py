"""
Video demo:
    
Launch it like this:
    - python3 examples/demo_video.py video_folder 
if you do not provide any folder it will create random data on the fly
"""
import time
import itertools
import tqdm
import cv2
from examples.video_stream_dataset import VideoLoader 
from torchvision.utils import make_grid


def make_grid_base(batch):
    """
    U8 Tensors B,3,H,W or B,1,H,W
    """
    batch_size = len(batch)
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    return make_grid(batch, nrows=nrows).detach().cpu().permute(1, 2, 0).numpy()


def read_dataset(path="", batch_size=4, num_workers=2, num_batches=400,
        viz=True, backend='scikit'):
    dataloader = VideoLoader(path, batch_size, num_workers, backend=backend) 
    start = time.time()
    for batch in tqdm.tqdm(itertools.islice(dataloader, num_batches), total=num_batches):
        if viz:
            for t in range(len(batch)):
                img = make_grid_base(batch[t])
                print(img.shape)
                # cv2.imshow('img', img[..., ::-1])
                # cv2.waitKey(5)
        else:
            time.sleep(0.1)
    end = time.time()
    print('duration: ', end-start)


if __name__ == '__main__':
    import fire
    fire.Fire(read_dataset)
