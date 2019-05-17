"""Dataset class template

We can specify '--dataset_mode faces' to use this dataset.
The class name needs to stay <Dataset_mode>Dataset.py
We need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
import torchvision.transforms as transforms
from PIL import Image
import re
import random


class FacesDataset(BaseDataset):
    """Our big wrapper custom dataset."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--cuhk_dir', type=str, default='cuhk', help='Sub-directory with cuhk files')
        parser.add_argument('--preprocess_dir', type=str, default='', help='Sub-directory of train/ or test/ with the files to use')
        parser.add_argument('--colorferet_dir', type=str, default='colorferet', help='Sub-directory with colorferet files')
        parser.add_argument('--iiitd_dir', type=str, default='fill me in', help='Sub-directory with iiitd files')
        return parser

    @staticmethod
    def make_cuhk_pairs_dataset(dir, max_dataset_size=float("inf")):
        image_pairs = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        sketch_regex = r'(\w)[12]?(-[^a-z]*)-sz1\.jpg'
        pic_regex =    r'(\w)[12]?(-[^a-z]*)\.jpg'
        sketch_paths = {}
        pic_paths = {}

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)

                match = re.search(sketch_regex, fname.lower())
                if match:
                    key = match[1] + match[2]
                    if key in pic_paths:
                        pic_path = pic_paths[key]
                        image_pairs.append((path, pic_path))
                        del pic_paths[key]
                        continue
                    sketch_paths[key] = path
                    continue

                match = re.search(pic_regex, fname.lower())
                if match:
                    key = match[1] + match[2]
                    if key in sketch_paths:
                        sketch_path = sketch_paths[key]
                        image_pairs.append((sketch_path, path))
                        del sketch_paths[key]
                        continue
                    pic_paths[key] = path
        return image_pairs[:min(max_dataset_size, len(image_pairs))]

    def make_iiitd_pairs_dataset(dir, max_dataset_size=float("inf")):
        image_pairs = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        sketch_paths = {}
        pic_paths = {}

        for root, dirs, files in sorted(os.walk(dir)):
            for f in files:
                path = os.path.join(root, f)

                if 'semi-forensic' in path:
                    key = 'semi-forensic' + f.lower()
                    if 'sketch' in path: # sketch
                        if key in pic_paths:
                            pic_path = pic_paths[key]
                            image_pairs.append((path, pic_path))
                            del pic_paths[key]
                            continue
                        sketch_paths[key] = path
                        continue
                    else: # image
                        if key in sketch_paths:
                            sketch_path = sketch_paths[key]
                            image_pairs.append((sketch_path, path))
                            del sketch_paths[key]
                            continue
                        pic_paths[key] = path
                elif 'viewed' in path:
                    if 'sketch' in path: # sketch
                        key = 'viewed' + f[1:].lower() if f[0] == 's' else 'viewed' + f.lower()
                        if key in pic_paths:
                            pic_path = pic_paths[key]
                            image_pairs.append((path, pic_path))
                            del pic_paths[key]
                            continue
                        sketch_paths[key] = path
                        continue
                    else: # image
                        key = 'viewed' + f[1:].lower() if f[0] == 'p' else 'viewed' + f.lower()
                        if key in sketch_paths:
                            sketch_path = sketch_paths[key]
                            image_pairs.append((sketch_path, path))
                            del sketch_paths[key]
                            continue
                        pic_paths[key] = path
        return image_pairs[:min(max_dataset_size, len(image_pairs))]

    def make_colorferet_pairs_dataset(dir, max_dataset_size=float("inf")):
        image_pairs = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        sketch_paths = {}
        pic_paths = {}

        for root, dirs, files in sorted(os.walk(dir)):
            for f in files:
                path = os.path.join(root, f)

                key = f[:5]
                if 'sketch' in path: # sketch
                    if key in pic_paths:
                        pic_path = pic_paths[key]
                        image_pairs.append((path, pic_path))
                        del pic_paths[key]
                        continue
                    sketch_paths[key] = path
                    continue
                else: # image
                    if key in sketch_paths:
                        sketch_path = sketch_paths[key]
                        image_pairs.append((sketch_path, path))
                        del sketch_paths[key]
                        continue
                    pic_paths[key] = path
        return image_pairs[:min(max_dataset_size, len(image_pairs))]

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_cuhk_images = os.path.join(opt.dataroot, opt.cuhk_dir, opt.phase, opt.preprocess_dir) # get the image directory (eg ./datasets/CUHK/train)
        self.dir_iiitd_images = os.path.join(opt.dataroot, opt.iiitd_dir, opt.phase) # get the image directory (eg ./datasets/iiitd/train)
        self.dir_colorferet_images = os.path.join(opt.dataroot, opt.colorferet_dir, opt.phase) # get the image directory (eg ./datasets/colorferet/train)
        self.image_pair_paths = sorted(FacesDataset.make_cuhk_pairs_dataset(self.dir_cuhk_images, opt.max_dataset_size) +
                FacesDataset.make_iiitd_pairs_dataset(self.dir_iiitd_images, opt.max_dataset_size) +
                FacesDataset.make_colorferet_pairs_dataset(self.dir_colorferet_images, opt.max_dataset_size))
        # assert(self.opt.load_size >= self.opt.crop_size)

        self.input_nc = 3 # if self.opt.direction == 'photo2sketch' else 1 # TODO: put this back in and get it working
        self.output_nc = 3 # 1 if self.opt.direction == 'photo2sketch' else 3

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        sketch_path, photo_path = self.image_pair_paths[index]    # supposed to be strings
        sketch = Image.open(sketch_path).convert('RGB')           # needs to be a tensor (TODO: should be 'L' not 'RGB', but need to debug why that breaks)
        photo = Image.open(photo_path).convert('RGB')             # needs to be a tensor

        transform_params = get_params(self.opt, sketch.size)
        crop_start_x = random.randint(0, self.opt.load_size - self.opt.crop_size)
        crop_start_y = random.randint(0, self.opt.load_size - self.opt.crop_size)
        transform_params['crop_pos'] = (crop_start_x, crop_start_y)
        transform_params['flip'] = random.choice([True, False])
        sketch_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        photo_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        sketch = sketch_transform(sketch)
        photo = photo_transform(photo)

        return {'sketch': sketch, 'photo': photo, 'sketch_path': sketch_path, 'photo_path': photo_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_pair_paths)
