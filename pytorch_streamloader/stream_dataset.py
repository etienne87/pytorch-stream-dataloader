from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class StreamDataset(object):
    """Base Class for Stream Dataset user class.
    """

    def __init__(
        self,
        stream_files,
        worker_id=0,
        num_workers=1,
        num_streams=3,
        num_batches=100,
        num_tbins=1,
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.num_streams = num_streams
        self.num_batches = num_batches
        self.num_tbins = num_tbins
        self.streams = []
        self.select_partition(stream_files)
        self.stream_iter = 0
        self.iter = 0
        self.reset_streams()

    def __len__(self):
        return self.num_batches

    def reset_streams(self):
        raise Exception("Not Implemented")

    def reload_stream(self, stream_idx):
        raise Exception("Not Implemented")

    def select_partition(self, stream_files):
        """default partitioning"""
        print("num stream files: ", len(stream_files))
        num_workers = max(1, self.num_workers)
        num_files = len(stream_files) // num_workers
        start = self.worker_id * num_files
        end = (self.worker_id + 1) * num_files
        self.stream_files = stream_files[start:end]
        print("num stream files selected: ", len(self.stream_files))

    def __call__(self, array_dict):
        raise Exception("Not Implemented")
