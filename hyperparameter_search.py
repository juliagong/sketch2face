"""General-purpose training script for hyperparameter search, based on train.py.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

"""
import time
from options.hyperparameter_search_options import HyperparameterSearchOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import itertools 
import shutil
import os

def train(opt):   
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.stop_epoch+1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

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
        print('Training model with learning rate {}, batch size {}, and beta1 {} for {} epochs.'.format(lr, batch_size, beta1, opt.stop_epoch))
        opt.lr = lr
        opt.batch_size = batch_size
        opt.beta1 = beta1

        # Only save the last one
        opt.save_epoch_freq = opt.stop_epoch

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