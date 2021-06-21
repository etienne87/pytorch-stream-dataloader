from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import time
import numpy as np
import cv2

from torchvision.utils import make_grid
from pytorch_streamloader.text_stream_dataset import make_text_dataset


TEXTS = [
"".join([chr(j)+'_'+str(i)+";" for i in range(1000)])
for j in range(97, 97+27)
]

def test_simple_case():
    #1. generate & dump random text on files
    #2. stream them
    dataset = make_text_dataset(TEXTS)
    for batch in dataset:
        print('========')
        for i in range(len(batch)):
            x = "".join([chr(item) for item in batch[i]])
            print(x)

if __name__ == "__main__":
    import fire
    fire.Fire(test_simple_case)
