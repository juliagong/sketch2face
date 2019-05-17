#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMT Transformer experiment

Usage:
    nmt_experiment.py [options]
                 
Options:
    -h --help                               show this screen.
    --images_path=<string>                  path to images to compare                         

"""
import os
import re
import cv2
import numpy as np
from docopt import docopt
from skimage.measure import compare_ssim as ssim

def make_image_path_pairs(dir):
    image_pairs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    fake_regex = r'(.*)_fake_B.png'
    real_regex = r'(.*)_real_B.png'
    fake_paths = {}
    real_paths = {}

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            match = re.search(fake_regex, fname)
            if match:
                key = match[1]
                if key in real_paths:
                    real_path = real_paths[key]
                    image_pairs.append((path, real_path))
                    del real_paths[key]
                    continue
                real_paths[key] = path
                continue

            match = re.search(real_regex, fname)
            if match:
                key = match[1]
                if key in fake_paths:
                    fake_path = fake_paths[key]
                    image_pairs.append((fake_path, path))
                    del fake_paths[key]
                    continue
                real_paths[key] = path
    return image_pairs



def l1_distance(imageA, imageB):
    sq_error = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
    return sq_error / float(imageA.shape[0] * imageB.shape[1] * imageB.shape[2])

def l2_distance(imageA, imageB):
    sq_error = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
    return sq_error / float(imageA.shape[0] * imageB.shape[1] * imageB.shape[2])

def main():
    args = docopt(__doc__)
    images_path = args['--images_path']
    image_path_pairs = make_image_path_pairs(images_path)
    image_pairs = [(cv2.imread(fake), cv2.imread(real)) for fake, real in image_path_pairs]

    l1_distances = []
    l2_distances = []
    ssims = []
    for fake, real in image_pairs:
        l1_distances.append(l1_distance(fake, real))
        l2_distances.append(l2_distance(fake, real))
        ssims.append(ssim(fake, real, multichannel=True))

    avg_l1_dist = sum(l1_distances)/len(l1_distances)
    avg_l2_dist = sum(l2_distances)/len(l2_distances)
    avg_ssim = sum(ssims)/len(ssims)
    print("Average L1 dist between the {} image pairs: {}".format(len(l1_distances), avg_l1_dist))
    print("Average L2 dist between the {} image pairs: {}".format(len(l2_distances), avg_l2_dist))
    print("Average SSIM between the {} image pairs: {}".format(len(ssims), avg_ssim))

if __name__ == "__main__":
	main()