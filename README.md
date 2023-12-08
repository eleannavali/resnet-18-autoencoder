# Resnet 18 Autoencoder

## Overview
This project implements a ResNet 18 Autoencoder capable of handling input datasets of various sizes, including 32x32, 64x64, and 224x224. The architecture is based on the principles introduced in the paper [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) and the [Pytorch implementation of resnet-18 classifier](https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18).

> Note: The implementation follows the basic residual block architecture of ResNet, and no bottleneck architecture has been implemented.

## About ResNet-18

ResNet-18 represents a specific configuration within the Residual Network (ResNet) architecture, featuring a total of 18 layers. Its core structure is built upon basic residual blocks, where each block incorporates two convolutional layers complemented by batch normalization and Rectified Linear Unit (ReLU) activation functions. The essence of ResNet-18 lies in the creation of residual connections, wherein the output of these layers is added to the input, facilitating improved gradient flow during training.

## Encoder

PyTorch provides a ResNet-18 model primarily designed as a classifier trained on the ImageNet dataset. Leveraging this implementation, we devised the default version of our ResNet-18 encoder. This involved removing the final two layers—average pooling and the fully connected network—as well as the flattening procedure from PyTorch's model. This modification effectively isolates the encoder, omitting the classification component. Notably, this architecture performs optimally with datasets of sizes 64x64 and larger.

For datasets smaller than 64x64, the original residual network paper recommends specific adjustments. Our approach involves removing max pooling and layer 4. Given the absence of a max-pooling function, it is important to address the conv1x1 layer within the downsample sequential. Keeping conv1x1 layer could result in certain pixels remaining unreconstructed, manifesting as pure noise (verified by our experiments). In our methodology, we remedy this by substituting the conv1x1 layer with a conv3x3 layer. This strategic replacement ensures the proper reconstruction of all pixels.

## Decoder
The decoder mirrors the encoder's structure, striving to invert each layer. Given the non-invertible nature of max pooling, we employed bilinear upsampling with a scale factor of 2. This technique effectively achieves the desired size, emulating the inversion of max pooling. The same process applies to both the default and light versions of the network.

> Note: The `inplanes` parameter value in the decoder should match the number of channels in layer 1 of the encoder.

## Other Setup

- Different Input Sizes:
The implementation has undergone testing with input datasets of sizes 64x64 and 32x32. Scalability is facilitated by employing the default network for upscaling and the light network version for downscaling. No further adjustments are necessary for input sizes within the range of 28x28 to 224x224. However, for other sizes, additional research is recommended to validate the network's suitability or implement specific modifications.
- Different Number of Layers:
While the primary objective was to implement a ResNet-18 autoencoder, the architecture supports flexibility. We have extended our configuration to accommodate 34 layers in the default network and 20 layers in the light version. Expanding further involves altering the layer list provided as a parameter in both the encoder and the decoder. Maintaining the basic residual block simplifies the expansion process. However, implementing a bottleneck architecture akin to ResNet-50 requires independent implementation due to current limitations.

## Installation
All experiments were conducted using Python 3.10.9 within a virtual environment. We utilized virtualenv for project isolation, but other virtual environments like conda env are also suitable. The installation process is outlined below:
1. Clone the repository from GitHub:

    `git clone`

2. Create a virtual environment with Python 3.10 inside the 'src' folder:

    `cd src`
    `python -m venv env`

3. Activate the virtual environment:

    `source env/bin/activate`

4. Install the required dependencies:

    `pip3 install -r requirements.txt`


## Usage
To train the autoencoder on the CIFAR-10 dataset, capturing both visual results and metrics for each epoch, execute the following command using the `main.py` script:

    python main.py

## Results
The results obtained from the training process, incorporating the early stopping technique, are summarized below for the 16th epoch:

- Training Loss: 0.00093
- Validation Loss: 0.00064

Visual representations of the outcomes are available for both the training and testing datasets:

![train results](https://github.com/eleannavali/resnet-18-autoencoder/blob/main/img/16_epoch_from_train_dataset.png)
![test results](https://github.com/eleannavali/resnet-18-autoencoder/blob/main/img/16_epoch_from_test_dataset.png)



