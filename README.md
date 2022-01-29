# Stylegan3 detection study on AFHQ cat images
PRNU's study on StyleGAN3 images full-frame and patch-wise, based on AFHQ v2 dataset

## Dataset
Original AFHQv2 dataset → 15,000 high-quality images at 512 × 512 resolution - https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0

The fake dataset has been generated with StyleGAN3 on Google Colaboratory using AFHQv2 pre-trained model available on [NVIDIA website](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3).

To obtain a dataset containing cat images only, we filtered them using VGG16 (Convolutional Neural Network used in image recognition). [Link](https://github.com/anqitu/What-animal-are-you) 

Dataset size: 490 images

![Original](https://drive.google.com/file/d/1Om9Kav09WZY4PcnWEmbx4rHHowzS--ro/view?usp=sharing "Real image")

![Fake](https://drive.google.com/file/d/1_Hjc1m1EiyOwh7kM4f6QyB8DlbnZFjTa/view?usp=sharing "StyleGAN3 generated")

## Reference
_Identification of GAN images Fingerprints
Signal, Image and Video_ - Ali Akay Mert Akkor
Karras, Tero, et al. "Alias-free generative adversarial networks." _Advances in Neural Information Processing Systems_ (2021).
