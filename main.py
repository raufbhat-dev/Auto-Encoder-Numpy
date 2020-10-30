import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt 
import matplotlib
import importlib

from neuralnetnumpy import neuralnet 

NeuralNet = importlib.reload(neuralnet).NeuralNet

inputs, Y = sklearn.datasets.load_digits( n_class=10, return_X_y=True)
outputs = inputs

outputs = np.zeros((Y.size, Y.max()+1))
outputs[np.arange(Y.size),Y] = 1

inputs = inputs/255

np.random.seed(0)

network_arch = [{'layer_type':'input', 'size':inputs.shape[-1]},
                {'layer_type':'hidden', 'size':49, 'activation':'sigmoid'},
                {'layer_type':'output', 'size':inputs.shape[-1], 'activation':'leakyrelu'}]

loss_func = 'meansquared'
optimiser_method = 'momentum'
learning_rate = 0.001
epoch = 2000
partition = 1
mode = 'train'
loss_display_freq = 1

neural_net = NeuralNet(loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch,beta =0.9)

neural_net.createNetwork()
neural_net.train(inputs,inputs)


#*********************** Ploting Outputs ***************

neural_net.mode = 'test'
img_index=10
test_out_sample = neural_net.forwardPass(inputs[img_index], 1)

#actual
plt.gray() 
plt.imshow(inputs[img_index].reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

#regenrated
plt.imshow(test_out_sample.reshape(8,8),interpolation='bilinear', cmap=plt.cm.Greys_r) 
plt.show() 

#*********************** Ploting weights ***************

#layer1
fig=plt.figure(figsize=(10, 10))
columns = 7
rows = 7
for i in range(columns*rows):
    layer2 = neural_net.layers_list[0].w.T[i].reshape(8,8)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r)
plt.show()


#layer2
columns = 8
rows = 8
for i in range(columns*rows):
    layer2 = neural_net.layers_list[1].w.T[i].reshape(7,7)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(layer2,interpolation='bilinear', cmap=plt.cm.Greys_r)
plt.show()
