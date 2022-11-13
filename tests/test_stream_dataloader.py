"""
Unit tests for the stream dataloader
"""
import os
import platform
import numpy as np

TEST = 1

from pytorch_stream_dataloader.stream_dataloader import StreamDataLoader

if TEST:
    from pytorch_stream_dataloader.stream_dataset_v0 import StreamDataset
else:
    from pytorch_stream_dataloader.stream_dataset import StreamDataset
from pytorch_stream_dataloader.utils import split_batch_size, split_dataset_sizes
from collections import defaultdict

np.random.seed(0)


class DummyStream(object):
    def __init__(self, stream_num, num_tbins, data=None, max_len=None):
        self.pos = 0
        self.stream_num = stream_num 
        self.num_tbins = num_tbins
        if data == None:
            self.max_len = max_len 
            self.data = [i for i in range(self.max_len)]
        else:
            self.data = data
            self.max_len = len(data)

    def __len__(self):
        return self.max_len

    def __iter__(self):
        #print(f'Streaming from #{self.stream_num}')
        return self

    def __next__(self):
        if self.pos >= self.max_len:
            # print(f'{self.stream_num} {self.pos}/{self.max_len} StopIter')
            raise StopIteration
        max_pos = min(self.pos + self.num_tbins, self.max_len)
        data = self.data[self.pos:max_pos]
        self.pos = max_pos
        return data, self.stream_num




def collate_fn(data_list):
    frame_nums, stream_nums = zip(*data_list)
    return {"frame_num": frame_nums, "stream_num": stream_nums}


class TestClassMultiStreams(object):
    def setup_dataloader(self, stream_list, num_workers, batch_size, num_tbins,
            padding='zeros'):
        def iterator_fun(tmp):
            stream_num, max_len = tmp
            return DummyStream(stream_num, num_tbins, max_len=max_len)
        padding_value = ([-1] * num_tbins, -1)
        dataloader = StreamDataLoader(stream_list, iterator_fun, batch_size,
                num_workers, collate_fn, padding, padding_value)
        return dataloader

    def assert_all(self, dataloader, stream_list, num_tbins, batch_size):
        # WHEN
        streamed1 = defaultdict(list)
        for stream_num, stream_len in stream_list:
            stream = DummyStream(stream_num=stream_num, num_tbins=num_tbins,
                    max_len=stream_len)
            for pos, _ in stream:
                streamed1[stream_num] += [pos]

        streamed2 = defaultdict(list)
        batch_number = defaultdict(list)
        batch_iter = defaultdict(list)
        for j,batch in enumerate(dataloader):
            actual_batch_size = len(batch['stream_num'])
            # THEN: batch_size should always be equal to user defined batch_size
            assert batch_size == actual_batch_size

            for i in range(batch_size):
                stream_num = batch['stream_num'][i]
                if stream_num == -1:
                    continue
                streamed2[stream_num] += [batch['frame_num'][i]]
                batch_number[stream_num].append(i)
                batch_iter[stream_num].append(j)

        # print(sorted(streamed1.keys()), sorted(streamed2.keys()))
        # THEN: data is contiguous accross batches
        for k, v in batch_number.items():
            if len(set(v)) > 1:
                breakpoint()
            assert len(set(v)) == 1

        stream_dict = {k:v for k,v in stream_list}

        # THEN: ALL IS READ
        for k, v1 in streamed1.items():
            v2 = streamed2[k]
            if v1 != v2:
                breakpoint()
                print('stream_id: ', k, ' tbins: ', stream_dict[k], 'v1: ', v1, 'v2: ', v2)
            assert v1 == v2, " "+str(k)

    def test_zero_pad_num_streams(self, tmpdir):
        # num_streams%batch_size != 0 (2 worker)
        num_workers, num_streams, batch_size, num_tbins = 2, 10, 4, 3
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        # GIVEN
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)

        # THEN
        for i in range(3):
            self.assert_all(dataloader, stream_list, num_tbins, batch_size)

    def test_zero_pad_batch_size_greater_not_divisible(self, tmpdir):
        # batch_size > num_streams_per_worker
        # batch_size%num_workers != 0
        num_workers, num_streams, batch_size, num_tbins = 3, 13, 7, 5
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        # GIVEN
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)

        # THEN
        self.assert_all(dataloader, stream_list, num_tbins, batch_size)

    def test_zero_pad_batch_size_not_enough_streams(self, tmpdir):
        # batch_size > num_streams_per_worker
        # batch_size%num_workers != 0
        num_workers, num_streams, batch_size, num_tbins = 3, 2, 7, 5
        num_workers = 0 if platform.system() == 'Windows' else num_workers

        # GIVEN
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        dataloader = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)

        # THEN
        try:
            self.assert_all(dataloader, stream_list, num_tbins, batch_size)
            has_failed = False
        except BaseException:
            has_failed = True

        assert has_failed

    def test_split_size(self):
        stream_list = [i for i in range(3)]
        split_sizes = split_batch_size(batch_size=3, num_workers=2)
        stream_groups = split_dataset_sizes(stream_list, split_sizes)
        for stream_group, split_size in zip(stream_groups, split_sizes):
            assert len(stream_group) >= split_size

    def test_split_num_workers_greater_than_batch_size(self):
        stream_list = [i for i in range(10)]
        split_sizes = split_batch_size(batch_size=3, num_workers=6)
        stream_groups = split_dataset_sizes(stream_list, split_sizes)
        for stream_group, split_size in zip(stream_groups, split_sizes):
            assert len(stream_group) >= split_size

    def test_contains_empty_streams(self):
        np.random.seed(0)
        # GIVEN
        num_workers, num_streams, batch_size, num_tbins = 2, 13, 7, 5
        stream_list = [(i, num_tbins * np.random.randint(1, 4)) for i in range(num_streams)]
        for i in [1,4,7]:
            stream_list[i]= (i, 0)

        dl = self.setup_dataloader(stream_list, num_workers, batch_size, num_tbins)
        # THEN
        self.assert_all(dl, stream_list, num_tbins, batch_size)

    def test_single_stream_single_batch_size(self):
        # GIVEN
        num_workers, num_streams, batch_size, num_tbins = 1, 1, 1, 3
        stream_list = [(0,[1,2,3])]
        def iterator_fun(tmp):
            stream_num, data = tmp
            return DummyStream(stream_num, num_tbins, data=data)
        padding_value = ([0] * num_tbins, -1)
        dataloader = StreamDataLoader(stream_list, iterator_fun, batch_size,
                num_workers, collate_fn, 'zeros', padding_value)

        # THEN
        for i, batch in enumerate(dataloader):
            continue
        assert i == 0


