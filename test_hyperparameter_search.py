"""General-purpose testing script for hyperparameter search, based on train.py.
"""
import time
from options.hyperparameter_search_test_options import HyperparameterSearchTestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import itertools 
import shutil
import os
from test import test

if __name__ == '__main__':
    # TODO: do a directory walk, rather than expecting grid search arguments to be exactly the same. Or better yet, merge with hyperparameter_search.py


    opt = HyperparameterSearchTestOptions().parse()   # get training options
    base_name = opt.name
    base_checkpoints_dir = opt.checkpoints_dir
    original_dir_path = os.path.join(base_checkpoints_dir, base_name)

    opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, '{}-gridsearch'.format(base_name))

    for lr, batch_size, beta1 in itertools.product(opt.lrs, opt.batch_sizes, opt.beta1s):
        print('Testing model with learning rate {}, batch size {}, and beta1 {}.'.format(lr, batch_size, beta1))
        opt.lr = lr
        opt.batch_size = batch_size
        opt.beta1 = beta1

        new_name = '{}-lr={}-bs={}-b1={}'.format(base_name, lr, batch_size, beta1)
        new_dir_path = os.path.join(opt.checkpoints_dir, new_name)
        opt.name = new_name
        test(opt)