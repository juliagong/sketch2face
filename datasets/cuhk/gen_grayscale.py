import numpy as np
import os
from PIL import Image

test_dir = './test'
train_dir = './train'

image_list = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and 'sz1' not in f] + \
        [os.path.join(train_dir, f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)) and 'sz1' not in f]

for image in image_list:
    name = image.split('/')[-1].split('.')[0]
    img = Image.open(image).convert('L')
    if 'test' in image:
        img.save(os.path.join(test_dir, name + '_gray.jpg'))
    elif 'train' in image:
        img.save(os.path.join(train_dir, name + '_gray.jpg'))
