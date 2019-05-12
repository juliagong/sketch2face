"""Dataset class template

We can specify '--dataset_mode cuhk' to use this dataset.
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


class CuhkDataset(BaseDataset):
    """Our first custom dataset."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--max_rotation', type=float, default=10.0, help='max rotation for augmentation')
        parser.add_argument('--min_crop_scale', type=float, default=0.5, help='min crop scaling for augmentation')
        parser.add_argument('--max_crop_scale', type=float, default=1.0, help='max crop scaling for augmentation')
        parser.add_argument('--scale_height', type=int, default=250, help='output height after rrc')
        parser.add_argument('--scale_width', type=int, default=200, help='output width after rrc')
        parser.set_defaults(preprocess='rrc_rotate')  # specify dataset-specific default values
        return parser


    @staticmethod
    def make_pairs_dataset(dir, max_dataset_size=float("inf")):
        image_pairs = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        sketch_regex = r'(\w)2(-.*)-sz1\.jpg'
        pic_regex = r'(\w-[^a-z]*)\.jpg'
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
                    key = match[1]
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
        self.dir_images = os.path.join(opt.dataroot, opt.phase) # get the image directory (eg ./datasets/CUHK/train)
        self.image_pair_paths = sorted(CuhkDataset.make_pairs_dataset(self.dir_images, opt.max_dataset_size))
        # assert(self.opt.load_size >= self.opt.crop_size)

        # define the default transform function. We can use <base_dataset.get_transform>, or we can define our own custom transform function
        # self.transform = get_transform(opt)

        self.input_nc = 3 # if self.opt.direction == 'photo2sketch' else 1 # TODO: put this back in and get it working
        self.output_nc = 3 #1 if self.opt.direction == 'photo2sketch' else 3

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
        sketch_transform = get_transform_for_augmentation(self.opt, transform_params, grayscale=(self.input_nc == 1))
        photo_transform = get_transform_for_augmentation(self.opt, transform_params, grayscale=(self.output_nc == 1))

        sketch = sketch_transform(sketch)
        photo = photo_transform(photo)

        return {'sketch': sketch, 'photo': photo, 'sketch_path': sketch_path, 'photo_path': photo_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_pair_paths)

def get_transform_for_augmentation(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'rrc' in opt.preprocess:
        transform_list.append(transforms.RandomResizedCrop(opt.crop_size, scale=(opt.min_crop_scale, opt.max_crop_scale)))
        # transform_list.append(transforms.RandomResizedCrop((opt.scale_width, opt.scale_height), scale=(opt.min_crop_scale, opt.max_crop_scale)))
    if 'rotate' in opt.preprocess:
        transform_list.append(transforms.RandomRotation(opt.max_rotation, method))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)