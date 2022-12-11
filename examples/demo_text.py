"""
Text demo:

Launch it like this:
    - python3 examples/demo_text.py
"""
from examples.text_stream_dataset import TextLoader



def read_dataset(batch_size=4, num_workers=2):
    example_text = [
    "".join([chr(j)+'_'+str(i)+";" for i in range(10)])
    for j in range(97, 97+27)
    ]
    dataloader = TextLoader(example_text, batch_size, num_workers, 24)
    for batch in dataloader:
        print('========')
        for i in range(len(batch)):
            x = "".join([chr(item) for item in batch[i]])
            print(x)


if __name__ == '__main__':
    import fire;fire.Fire(read_dataset)
