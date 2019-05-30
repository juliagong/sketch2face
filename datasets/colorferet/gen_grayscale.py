import numpy as np
import os
from PIL import Image

test_color_dir = './test/color'
train_color_dir = './train/color'
test_gray_dir = './test/grayscale'
train_gray_dir = './train/grayscale'

image_list = [os.path.join(test_color_dir, f) for f in os.listdir(test_color_dir) if os.path.isfile(os.path.join(test_color_dir, f))] + \
        [os.path.join(train_color_dir, f) for f in os.listdir(train_color_dir) if os.path.isfile(os.path.join(train_color_dir, f))]

for image in image_list:
    name = image.split('/')[-1]
    img = Image.open(image).convert('L')
    if 'test' in image:
        img.save(os.path.join(test_gray_dir, name))
    elif 'train' in image:
        img.save(os.path.join(train_gray_dir, name))
