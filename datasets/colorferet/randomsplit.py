import os
import shutil
import numpy as np

color_dir = './color'
sketch_dir = './sketch'
color_test = './test/color'
sketch_test = './test/sketch'
color_train = './train/color'
sketch_train = './train/sketch'

image_list = [f for f in os.listdir(color_dir) if os.path.isfile(os.path.join(color_dir, f))]

for image in image_list:
    if np.random.rand(1) < 0.23: # test
        shutil.move(color_dir + '/' + image, color_test + '/' + image)
        shutil.move(sketch_dir + '/' + image[:5] + '.jpg', sketch_test + '/' + image[:5] + '.jpg')
    else: # train
        shutil.move(color_dir + '/' + image, color_train + '/' + image)
        shutil.move(sketch_dir + '/' + image[:5] + '.jpg', sketch_train + '/' + image[:5] + '.jpg')
