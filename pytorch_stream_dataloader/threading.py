"""
Threading Utils
"""
import threading
import queue

MISSING = object()


def join_in_queue(iterable, queue):
    for data in iterable:
        out_queue.put(data, timeout=MP_STATUS_CHECK_INTERVAL)
    out_queue.put(MISSING)


def join_data_threaded(args_iterables, target):
    threads = []
    data_queue = queue.Queue()
    for args in args_iterable:
        thread = threading.Thread(
            target=join_queue,
            args=(
                data_queue,
            ),
        )
        thread.start()
        threads.append(thread)

    join_memory_thread.daemon = True
    join_memory_thread.start()
    while True:
        data = data_queue.get(timeout=100000)
        if data is MISSING:
            raise StopIteration
        yield data


