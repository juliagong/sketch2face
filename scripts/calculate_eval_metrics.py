#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Credit to https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/ for ideas
"""
NMT Transformer experiment

Usage:
    nmt_experiment.py [options]
                 
Options:
    -h --help                               show this screen.
    --images_path=<string>                  path to images to compare                         
    --include_inception                     whether to calculate inception score (long & buggy right now)
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
                fake_paths[key] = path
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

def MAE(imageA, imageB): # same as torch.nn.L1Loss, which is put out
    abs_error = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
    mean_abs_error = abs_error / float(imageA.shape[0] * imageB.shape[1] * imageB.shape[2])
    return mean_abs_error

def RMSE(imageA, imageB): # Root-mean-squared error, equivalent of L2
    sq_error = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
    mse = sq_error / float(imageA.shape[0] * imageB.shape[1] * imageB.shape[2])
    return np.sqrt(mse)

def main():
    args = docopt(__doc__)
    images_path = args['--images_path']
    image_path_pairs = make_image_path_pairs(images_path)
    image_pairs = [(cv2.imread(fake), cv2.imread(real)) for fake, real in image_path_pairs]

    l1_distances = []
    l2_distances = []
    ssims = []
    for fake, real in image_pairs:
        l1_distances.append(MAE(fake, real))
        l2_distances.append(RMSE(fake, real))
        ssims.append(ssim(fake, real, multichannel=True))

    avg_l1_dist = sum(l1_distances)/len(l1_distances)
    avg_l2_dist = sum(l2_distances)/len(l2_distances)
    avg_ssim = sum(ssims)/len(ssims)
    
    print("Average mean average error (l1 loss) between the {} image pairs: {}".format(len(l1_distances), avg_l1_dist))
    print("Average root mean squared error (l2 loss) between the {} image pairs: {}".format(len(l2_distances), avg_l2_dist))
    print("Average SSIM between the {} image pairs: {}".format(len(ssims), avg_ssim))
    
    if (args['--include_inception']):
        from inception_distance import get_fid
        fakes = np.array([fake for fake, _ in image_pairs]).transpose((0, 3, 1, 2))
        reals = np.array([real for _, real in image_pairs]).transpose((0, 3, 1, 2))
        avg_inception = get_fid(fakes, reals)
        print("Frechet inception distance between the distributions: {}".format(avg_inception))

if __name__ == "__main__":
	main()