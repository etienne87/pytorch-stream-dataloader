# pytorch-streamloader

**The dataloader to stream videos or text or anything in temporally coherent batches.**

## What is it?

~With current implementation of iterable dataset I don't manage to stream several videos/ text/ audio in temporally coherent batches *with several workers*.
Here i provide a simple implementation of streaming with multiprocessing and pytorch.
This is mainly to get feedback and understand how to do this better/ simpler, but if you find this useful don't hesitate to give me feedback as well.~

EDIT: i now manage to make the same thing with the pytorch iterable dataset, it is very easy in fact (look at pytorch_iterable.py)

![](data/dataloader_figure.jpg)

## Text Example

A very simple example can be found in examples/text_dataset.py together with pytorch_stream_loader/text_stream_dataset.py

```
TEXTS = [
"".join([chr(j)+'_'+str(i)+";" for i in range(1000)])
for j in range(97, 97+27)
]
dataset = make_text_dataset(TEXTS)
for j, batch in enumerate(dataset):
    print('batch'+str(j)+': ')
    for i in range(len(batch)):
        x = "".join([chr(item) for item in batch[i]])
        print(x)
```
This will show: 
```
- batch1
a_0;a_1;a_2;
b_0;b_1;b_2;
c_0;c_1;c_2;
d_0;d_1;d_2;
e_0;e_1;e_2;
f_0;f_1;f_2;
g_0;g_1;g_2;
h_0;h_1;h_2;

- batch2
a_3;a_4;a_5;
b_3;b_4;b_5;
c_3;c_4;c_5;
d_3;d_4;d_5;
e_3;e_4;e_5;
f_3;f_4;f_5;
g_3;g_4;g_5;
h_3;h_4;h_5;

- batch3
a_6;a_7;a_8;
b_6;b_7;b_8;
c_6;c_7;c_8;
d_6;d_7;d_8;
e_6;e_7;e_8;
f_6;f_7;f_8;
g_6;g_7;g_8;
h_6;h_7;h_8;
...
```
You notice that every row is a coherent sequence (marked by the letter and timestep number for sake of example). 
And that this continuity extends accross batches.

### How to make this streaming of text?
```
def make_text_dataset(
    words,
    num_iter=10,
    num_tbins=80,
    num_workers=1,
    batchsize=8,
    max_frames=100,
    start_epoch=0,
):
    array_dims = (num_tbins, 1)
    make_env = partial(
        TextStreams,
        stream_files=words,
        num_batches=num_iter,
        max_frames=max_frames,
        num_tbins=num_tbins,
    )
    dataset = MultiStreamer(
        make_env,
        array_dims,  
        batchsize=batchsize,
        max_q_size=4,
        num_workers=num_workers,
        collate_fn=collate_fn,
        epoch=start_epoch,
        dtype=np.uint8,
        main_thread=0,
        pin_memory=0,
    )

    return dataset
```
Here we give the dataset generator to the MultiStreamer, which is the main dataloader instanciating the threads (think of it as the Pytorch DataLoader). 
Here Each worker_id has its own "TextStreams" class that delivers "micro-batches" (Batchsize/num_workers, Tbins, 1) that are collated in the multistreamer for which you can also pass a custom collate function.

### How to write your own 

Here an example of the text streamer

``` 
class TextStreams(StreamDataset):
    def __init__(
        self,
        stream_files=[],
        worker_id=0,
        num_workers=1,
        num_streams=3,
        num_batches=100,
        num_tbins=1,
        epoch=0,
        **kwargs
    ):
        self.max_frames = 1000
        self.epoch = epoch
        self.stream_files = stream_files
        super(TextStreams, self).__init__(
            stream_files, worker_id, num_workers, num_streams, num_batches, num_tbins
        )

    def reset_streams(self):
        self.streams = []
        for i in range(self.num_streams):
            self.streams.append(TextStream(self.stream_files[self.stream_iter], self.num_tbins))
            self.stream_iter += 1

    def reload_stream(self, idx):
        self.streams[idx].reload(self.words[self.stream_iter])
        self.stream_iter = (self.stream_iter + 1) % len(self.stream_files)

    def __call__(self, arrays_dic):
        batchsize, tbins = arrays_dic["data"].shape[:2]
        assert len(self.streams) == batchsize
        mask = np.zeros((batchsize), dtype='u8')
        filenames = []
        times = []
        self.iter += 1
        for i, stream in enumerate(self.streams):
            filenames_i = []
            times_i = []
            frame = next(stream)
            while frame is None:
                self.reload_stream(i)
                frame = next(self.streams[i])
            mask[i] = self.streams[i].iter > 1
            arrays_dic["data"][i, :len(frame)] = frame[:,None]
        return {"resets": [mask]}
``` 

your dataset needs to instanciate in a mandatory for the multistreamer 1 function:

``` 
def __call__(self, arrays_dic):
```
and fill every item of this dictionary with items of shape compatible with the dictionary "array_dims" that you gave to Multistreamer.

I provide an example of class that you can use to derive from: the StreamDataset which contains mandatory functions to re-implement.

- reset_streams: build all streaming objects
- reload_stream: load the next file for one of your streaming object which has run out of data to stream__
- call: fill current dictionary of data.




## Video Example:

You can run the example/video_dataset.py on any folder containing .mp4! 
This should show you a grid of several videos being read at the same time and delivered with "minimal" latency to pytorch GPU. (well that is the idea at least). This indicates a timing around 1 ms to deliver a batch (because the main process is showing the frames and takes time on its own).

![](data/example_video.gif)

## Virtual Camera in front of Planar Image Example:

This example showcases that you can do completely procedural data streaming to your network (with data-parallelism).

## Scrapping Articles from internet and streaming them

COMING SOON


## Runtimes

COMING SOON



## Installation

COMING SOON


