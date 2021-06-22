"""
demos
"""
import time
import itertools
import tqdm
from examples.video_stream_dataset import VideoLoader as TorchVideoLoader


def read_dataset(path, batch_size=4, num_workers=2, num_batches=100):
    dataloader = TorchVideoLoader(path, batch_size, num_workers)
    start = time.time()
    for batch in tqdm.tqdm(itertools.islice(dataloader, num_batches), total=num_batches):
        continue
    end = time.time()
    print('duration: ', end-start)




if __name__ == '__main__':
    import fire;fire.Fire()
