import numpy as np
import importlib

from neuralnetnumpy import layer, optimiser, loss

layer = importlib.reload(layer)
optimiser = importlib.reload(optimiser)
loss = importlib.reload(loss)

class NeuralNet:
    def __init__(self, loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch,**kwargs):
        self.loss_func = loss_func
        self.optimiser = optimiser_method
        self.epoch_count = epoch
        self.partition_size = partition_size 
        self.mode = mode
        self.layers_list = []
        self.loss  = loss.RegularisedLoss(loss_func, kwargs.get('reularisation',None), kwargs.get('gamma',0))
        self.network_arch = network_arch
        if optimiser_method == 'momentum':
            momentum = True
            self.optimiser = optimiser.GradientDescent(momentum, learning_rate,beta = kwargs.get('beta',0.9))
        else:
            momentum = False
            self.optimiser = optimiser.GradientDescent(momentum, learning_rate)
                    
    def createNetwork(self):
        network_layers = []
        for index, _layer in  enumerate(self.network_arch):
                if _layer['layer_type'] != 'input':
                    self.layers_list.append(layer.Layer(self.network_arch[index-1]['size'],_layer['size'], _layer['activation'],_layer['layer_type']))

    def forwardPass(self, inputs, outputs):
        layer_out = inputs
        for _layer in self.layers_list:
            layer_out = _layer(inputs)
            inputs = layer_out
            self.loss.accumulateRegularisedLoss(_layer.w)
        if self.mode.lower() == 'train':
            self.loss.getLoss()(layer_out,outputs)
            self.loss.getLossTotal()
        elif self.mode.lower() == 'test':
            return layer_out
    
    def backProp(self, inputs):
        upstream_gradient = self.loss.loss_derivative
        for index, _layer in enumerate(reversed(self.layers_list)):
            if _layer.layer_type == 'output':
                upstream_gradient =  np.multiply(_layer.activation_derivative, upstream_gradient)
                upstream_gradient_w =  np.matmul(self.layers_list[len(self.layers_list)-2].y_activation.T, upstream_gradient) 
            if _layer.layer_type == 'hidden':
                upstream_gradient =  np.matmul(upstream_gradient, self.layers_list[len(self.layers_list) -index].w.T)
                upstream_gradient = np.multiply(upstream_gradient,_layer.activation_derivative)
                if (len(self.layers_list)-index-1) != 0:
                    upstream_gradient_w = np.matmul(self.layers_list[len(self.layers_list) -index -2].y_activation.T,upstream_gradient)
                else:
                    upstream_gradient_w = np.matmul(inputs.T,upstream_gradient)
            upstream_gradient_b = np.sum(upstream_gradient,axis=0).T
            self.optimiser(_layer, upstream_gradient_w, upstream_gradient_b)
    
        for _layer_ in self.layers_list:
            _layer_.w = _layer_.w + _layer_.w_delta + self.loss.regularisedLossGradient(_layer_.w)
            _layer_.b = _layer_.b + _layer_.b_delta + self.loss.regularisedLossGradient(_layer_.b)
    
    def train(self, inputs, outputs):
        inputs = np.array_split(inputs, self.partition_size)
        Y =  np.array_split(outputs, self.partition_size)
        for i in range(self.epoch_count):
            for index, (inp_batch, out_batch) in enumerate(zip(inputs, Y)):  
                self.forwardPass(inp_batch, out_batch)
                self.backProp(inp_batch)
                print('Epoch:{}  batch: {}/{}  Overall Loss : {}'.format(i+1, index+1, self.partition_size, self.loss.loss_total), end='\r', flush=True)
            print("",end="\n")
                
