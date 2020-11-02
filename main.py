
import numpy as np
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt 
import importlib

from autoencodernumpy import autoencoder 
AutoEncoder = importlib.reload(autoencoder).AutoEncoder

inputs, Y = sklearn.datasets.load_digits( n_class=10, return_X_y=True)

inputs = sklearn.preprocessing.scale(inputs)

np.random.seed(0)

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1],'network_part':'encoder'},
                {'layer_type':'hidden', 'size':49, 'activation':'sigmoid','network_part':'encoder'},
                {'layer_type':'output', 'size':inputs.shape[-1], 'activation':'leakyrelu','network_part':'decoder'}]

loss_func = 'meansquared'
optimiser_method = 'momentum'
learning_rate = 0.001
epoch = 50000
partition = 1
mode = 'train'
loss_display_freq = 1

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
plt.imshow(inputs[img_index].reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

#regenrated
print('Reconstructed image for index:{}'.format(img_index))
plt.imshow(test_out_sample.reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 


#*********************** Ploting weights ***************

print('Weights')
#encoder
fig = plt.figure(figsize=(10, 10))
for index, _layer in enumerate(encoder_layers):
    rows = columns = 7
    print('Encoder layer: {}'.format(index))
    image_shape = 8
    for i in range(columns*rows):
        layer2 = _layer.w.T[i].reshape(image_shape,image_shape)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r)
    plt.show()

#decoder
fig = plt.figure(figsize=(10, 10))
for index, _layer in enumerate(decoder_layers):
    rows = columns = 8
    print('Decoder layer: {}'.format(index))
    image_shape = 7
    for i in range(columns*rows):
        layer2 = _layer.w.T[i].reshape(image_shape,image_shape)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r)
    plt.show()
