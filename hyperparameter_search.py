"""General-purpose training script for hyperparameter search, based on train.py.

Example:
    Train a pix2pix model:
        python hyperparameter_search.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

"""
import time
from options.hyperparameter_search_options import HyperparameterSearchOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import itertools 
import shutil
import os
from train import train

if __name__ == '__main__':
    opt = HyperparameterSearchOptions().parse()   # get training options
    base_lr = opt.lr
    base_batchsize = opt.batch_size
    base_beta1 = opt.beta1
    base_name = opt.name
    base_checkpoints_dir = opt.checkpoints_dir
    original_dir_path = os.path.join(base_checkpoints_dir, base_name)

    opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, '{}-gridsearch'.format(base_name))
    if not os.path.isdir(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)

    for lr, batch_size, beta1 in itertools.product(opt.lrs, opt.batch_sizes, opt.beta1s):
        print('Training model with learning rate {}, batch size {}, and beta1 {} for {} epochs.'.format(lr, batch_size, beta1, opt.max_epochs))
        opt.lr = lr
        opt.batch_size = batch_size
        opt.beta1 = beta1

        # Only save the last one
        # opt.save_epoch_freq = opt.max_epoch

        new_name = '{}-lr={}-bs={}-b1={}'.format(base_name, lr, batch_size, beta1)
        new_dir_path = os.path.join(opt.checkpoints_dir, new_name)
        if os.path.isdir(original_dir_path):
            try:
                shutil.copytree(original_dir_path, new_dir_path)
            except shutil.Error as e:
                print('Directory not copied. Error: %s' % e)
            except OSError as e:
                print('Directory not copied. Error: %s' % e)
        opt.name = new_name
        train(opt)