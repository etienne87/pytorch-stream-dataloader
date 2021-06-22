"""
Utils functions
"""
import random


def split_batch_size(batch_size, num_workers):
    """Returns a list of batch_size

    Args:
        batch_size (int): total batch size
        num_workers (int): number of workers
    """
    num_workers = min(num_workers, batch_size)
    split_size = batch_size // num_workers
    total_size = 0
    split_sizes = [split_size] * (num_workers - 1)
    split_sizes += [batch_size - sum(split_sizes)]
    return split_sizes


def split_dataset_sizes(stream_list, split_sizes):
    """Splits with different sizes

    Args:
        stream_list (list): list of stream path
        split_sizes (list): batch size per worker
    """
    out = []
    start = 0
    total = sum(split_sizes)
    for split_size in split_sizes[:-1]:
        num = int(split_size / total * len(stream_list))
        end = start + num
        out.append(stream_list[start:end])
        start = end
    out.append(stream_list[start:])
    return out


def resample_to_batch_size(stream_list, batch_size):
    """Resamples list to fit batch_size iterators

    Args:
        stream_list (list): list of streams
        batch_size (int): batch size
    """
    stream_list = random.sample(stream_list, len(stream_list)) +\
        random.choices(stream_list, k=batch_size - len(stream_list) % batch_size)
    return stream_list
