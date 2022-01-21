# ComputerVision
This repository  aims at realized easy network architecture

## Colorize picture with Autoencoder architecture

A simple usage of UNet, UNet++, UNet+++ is to color the gray scale image, input a gray scale image with RGB image as label, the network will leran to color the image.

![UNet with normalize](https://github.com/DongDong-Zoez/ComputerVision/blob/main/Image%20Segmentation/UNet/colorNet_n.png)

## Image Segmentation

Convolutional autoencoder architecture are widely applied to Image Segmentation, input a image with label the boundary, the network will learn to partition the image.

![UNet++ segmentation](https://github.com/DongDong-Zoez/ComputerVision/blob/main/Image%20Segmentation/UNet%2B%2B/segment.png)

## Objective Detection

We apply state-of-the-art objective detection technique YOLOX on the city potholes dataset

<img src="https://github.com/DongDong-Zoez/ComputerVision/blob/79875733078b656bfa8f3901e2f47fb31af4e853/Objective%20Detection/YOLOX_potholes.jpg" width="400" height="300" alt="图片描述文字"/><img src="https://github.com/DongDong-Zoez/ComputerVision/blob/a026060b111a853e00753f8b9fac9649b88dba27/Objective%20Detection/YOLOX_potholes1.jpg" width="400" height="300" alt="图片描述文字"/>

## GAN

DCGAN is the most widely applied GAN to generate image, we used DCGAN with four convolution layers

<div style="width: 543px;height: 542px;overflow: hidden;">
    <img src="https://github.com/DongDong-Zoez/ComputerVision/blob/f5ed25e5be0f85c814135082c369b0d0802e7e9d/GAN/DCGAN/video.gif" style="margin-top: -87px; margin-left: -98px;">
</div>

## Super-Resolution

64*64 to 256*256

- Fake HR: Generate image
- HR: High resolution image
- LR: Low resolution image

![SRGAN](https://github.com/DongDong-Zoez/ComputerVision/blob/d2c75669b266404510c88870c80c922ef8790dd9/GAN/SRGAN.png)
