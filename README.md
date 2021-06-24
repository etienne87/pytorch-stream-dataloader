# Pytorch-Stream-Dataloader

**A light wrapper dataloader to stream videos or text or anything in temporally coherent batches for recurrent networks.**

![alt_text](https://cdn.futura-sciences.com/buildsv6/images/wide1920/0/e/2/0e209aae81_128445_fs-theatre-optique.jpg)


# Install

```pip install pytorch-stream-dataloader==1.0```

# What is it?

With current implementation of iterable dataset I don't manage to stream several videos / text / audio in temporally coherent batches **with several workers**. What happens with batch_size=X and num_workers=X is that you receive in any order the batches coming from various workers, **one after the other**. There is no automatic collation of the data to stream one unified batch.
Here i provide a simple implementation of unified batch of streams, by implementing a wrapper around Pytorch's IterableDataset.
This is mainly to get feedback and understand how to do this better / simpler, but if you find this useful don't hesitate to give me feedback as well.

**EDIT 21-06-2020**: i now manage to make the same thing with the pytorch iterable dataset, following https: // medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

**EDIT 21-06-2021**: i now use iterable a bit differently, i ask every IterableDataset to retrieve the worker's id.

With Pytorch Iterable Dataset that returns the worker's id, you can also avoid re-concatenating all the data & simply have different RNNs indexed by the worker's id. This way you do not even need the StreamDataLoader's logic, only the StreamDataset class (and write your own iterator).

Example:

```
class MyMagnificoIterable(IterableDataset):
    ...
    def __iter__(self):
        ...

        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        yield my_data, worker_id
...
ds = MyMagnificoIterable(files)  # make sure this yields the data AND the worker's id.

dataloader = torch.utils.DataLoader(ds, batch_size=None, num_workers=whatever)
for batch, worker_id in dataloader:
    the_good_rnn = my_rnns[worker_id]
    y = the_good_rnn(batch)
    ...
```
The difference with the StreamDataset is that it handles automatically streaming over several iterables simultaneously, so if you use it, you only have to write the iterator over one stream only. Think of it as "stream grouper" of iterables that you do not need to write yourself. StreamDataset replaces ChainDataset and also collates the data from several iterables. 

When using the StreamDataset, depending on your batch size and the number of workers, the workload is automatically dispatched accross instances of this class, so you do not have to worry about data partitioning. All the streams are read **at least once**, you have the choice for batch "rows" that have no longer any data to read to either stream: 
- data that you have read before
- padding data that your trainer can recognize as dummy.

Generally you would want to stream for rnn:
- current batch of data
- if the stream has just started (useful to reset the memory at this example)
- some metadata for kpi computation...

# Schematic to understand DataLoading for RNN:

![](data/dataloader_figure.jpg)

# Text Example

A very simple example can be found in examples/demo_text.py together with examples/text_stream_dataset.py

Here an example of the text stream iterator:
```
class TextStream(object):
    def __init__(self, text, tbins):
        self.text = np.fromstring(text, dtype=np.uint8)
        self.iter = 0
        self.tbins = tbins

    def __iter__(self):
        for i in range(0, len(self.text), self.tbins):
            data = self.text[i:i+self.tbins]
            #pad to tbins
            frame = np.zeros((1, self.tbins), dtype=np.float32)
            frame[0,:len(data)] = data
            yield (torch.from_numpy(frame), 0)
```
That's it! You just have to create your own iterator, that can be constructed
Here is how you would give this class to the StreamDataset:

```
def collate_fn(data_list):
    texts, _ = zip(*data_list)
    texts = torch.cat(texts)
    return texts


class TextLoader(StreamDataLoader):
    def __init__(self, texts, batch_size, num_workers, tbins=5):
        def iterator_fun(text): #define a lambda to open ONE file
            return TextStream(text, tbins)
        dataset = StreamDataset(texts, iterator_fun, batch_size, "data", None) # collection of iterables
        super().__init__(dataset, num_workers, collate_fn) # stream-dataloader wrapper
```

Here we give the dataset to the StreamDataloader, which is a small wrapper around Pytorch's DataLoader. All it does is receive batches from the IterableDataset "StreamDataset" and worker ids and collate them as it receives them from the Pytorch's DataLoader.
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
You notice that every row is a coherent sequence(marked by the letter and timestep number for sake of example).
And that this continuity extends accross batches.


# Video Example:

You can run the example/video_dataset.py on any folder containing .mp4!
This should show you a grid of several videos being read at the same time and delivered with "minimal" latency to pytorch GPU. (well that is the idea at least). This indicates a timing around 1 ms to deliver a batch(because the main process is showing the frames and takes time on its own).

![](data/example_video.gif)
