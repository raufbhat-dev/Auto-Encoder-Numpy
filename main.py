import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import struct
import importlib
import os

from autoencodernumpy import autoencoder, denoisingautoencoder
from utility import Utility 

DenoisingAutoEncoder = importlib.reload(denoisingautoencoder).DenoisingAutoEncoder
AutoEncoder = importlib.reload(autoencoder).AutoEncoder

np.random.seed(1)

inputs = Utility.readIDX('/mnt/ebs-1/rauf_bhat/git_repo_rauf/DataSets/train-images-idx3-ubyte')
lables = Utility.readIDX('/mnt/ebs-1/rauf_bhat/git_repo_rauf/DataSets/train-labels-idx1-ubyte')
inputs = inputs.reshape(inputs.shape[0],28*28)

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1],'network_part':'encoder'},
                {'layer_type':'hidden', 'size':20, 'activation':'sigmoid','network_part':'encoder'},
                {'layer_type':'output', 'size':inputs.shape[-1], 'activation':'leakyrelu','network_part':'decoder'}]

loss_func = 'meansquared'
optimiser_method = 'momentum'
learning_rate = 0.001
epoch = 10
partition = 1000
mode = 'train'
loss_display_freq = 1
percentage = 0.2
                                          
inputs = Utility.dataScale(inputs)

auto_encoder = AutoEncoder(loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch,beta =0.9)

auto_encoder.createNetwork()

auto_encoder.train(inputs,inputs)

encoder_layers = auto_encoder.encoder_layer_list
decoder_layers = auto_encoder.decoder_layer_list

#*********************** Ploting Outputs ***************
auto_encoder.mode = 'test'

img_index = 1
test_out_sample = auto_encoder.forwardPass(inputs[img_index], 1)

#actual
print('Actual image for index:{}'.format(img_index))
plt.gray() 
plt.imshow(inputs[img_index].reshape(28,28),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

#regenrated
print('Reconstructed image for index:{}'.format(img_index))
plt.imshow(test_out_sample.reshape(28,28),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

#*********************** Ploting weights ***************
Utility.plotWeights(encoder_layers, save = False)
Utility.plotWeights(decoder_layers, save = False)

#Denoising AutoEncoder

auto_encoder = DenoisingAutoEncoder(loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch,beta =0.9)

auto_encoder.createNetwork()

inputs = auto_encoder.addGaussianNoise(inputs,0.2)

auto_encoder.mode = 'train'
auto_encoder.train(inputs,inputs)

encoder_layers = auto_encoder.encoder_layer_list
decoder_layers = auto_encoder.decoder_layer_list

#*********************** Ploting weights ***************
Utility.plotWeights(encoder_layers, save = False)
Utility.plotWeights(decoder_layers, save = False)
