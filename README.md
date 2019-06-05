# sketch2face: Conditional Generative Adversarial Networks for Transforming Face Sketches into Photorealistic Images
Generation of color photorealistic images of human faces from their corresponding grayscale sketches, building off of code from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Abstract
In this paper, we present a conditional GAN image translation model for generating realistic human portraits from artist sketches. We modify the existing pix2pix model by introducing four variations of an iterative refinement (IR) model architecture with two generators and one discriminator, as well as a model that incorporates spectral normalization and self-attention into pix2pix. We utilize the CUHK Sketch Database and CUHK ColorFERET Database for training and evaluation. The best-performing model, both qualitatively and quantitatively, uses iterative refinement with L1 and cGAN loss on the first generator and L1 loss on the second generator, likely due to the first-stage sharp image synthesis and second-stage image smoothing. Most failure modes are reasonable and can be attributed to the small dataset size, among other factors. Future steps include masking input images to facial regions, trying other color spaces, jointly training a superresolution model, using a colorization network, learning a weighted average of the generator outputs, and gaining control of the latent space of generated faces.

## Directory Guide
Relevant folders that were significantly modified during the course of this project are:

[checkpoints](checkpoints/) contains model logs and training options.

[data](data/) contains the data classes used for handling the data that interface with the models.

[datasets](datasets/) contains the ColorFERET and CUHK datasets used for training and testing the models.

[facenet-pytorch](facenet-pytorch/) contains the cloned GitHub from [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch?fbclid=IwAR0rPyB1nMOY12Z4VgBCn89Z4qrC7xoS_Z0wTN9dx0YXa44ZJzlm69muq8s) and the implemented FaceNet evaluation metrics for the model.

[models](models/) contains the model classes for the [baseline model](models/sketch2facebaseline_model.py), [color iterative refinement models](models/sketch2face_model.py), [grayscale iterative refinement model](models/sketch2faceg_model.py), and modified implementations for [spectral normalization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan) and [self-attention from SAGAN](https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py).

[options](options/) contains training and testing options, as well as custom model options for the baseline and the iterative refinement models.

[results](results/) contains the test output images for all 294 samples for each of the models implemented.

[scripts](scripts/) contains the script to run evaluation metrics for L1, L2 distance and SSIM.
