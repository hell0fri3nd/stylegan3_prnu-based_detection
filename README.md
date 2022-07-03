# Stylegan3 detection study on AFHQ cat images
PRNU's study on StyleGAN3 images full-frame and patch-wise, based on AFHQ v2 dataset

## Dataset
Original AFHQv2 dataset → 15,000 high-quality images at 512 × 512 resolution - https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0

The fake dataset has been generated with StyleGAN3 on Google Colaboratory using AFHQv2 pre-trained model available on [NVIDIA website](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3).

To obtain a dataset containing cat images only, we filtered them using VGG16 (Convolutional Neural Network used in image recognition). [Link](https://github.com/anqitu/What-animal-are-you) 

Dataset size: 490 images

## Results
Results can be seen at the following [link](https://docs.google.com/presentation/d/1cngmtWVaQjFE_XlxJ8hiBCPNrrIruyq4EXPnvQwfwR4/edit?usp=sharing)

## Project folder

For the whole project folder & additional material check [here](https://drive.google.com/drive/folders/1i1uSvDs16jYILxBVMQmuoGOg3TD-tp8R?usp=sharing) 

## Reference
_Identification of GAN images Fingerprints
Signal, Image and Video_ - Ali Akay Mert Akkor

Karras, Tero, et al. "Alias-free generative adversarial networks." _Advances in Neural Information Processing Systems_ (2021).
