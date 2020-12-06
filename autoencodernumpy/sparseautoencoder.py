import numpy as np
import importlib

from autoencodernumpy import autoencoder
AutoEncoder = importlib.reload(autoencoder).AutoEncoder

class SparseAutoEncoder(AutoEncoder):
    def __init__(self,loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch, roh, **kwargs):
        super().__init__(loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch, **kwargs)
        self.roh = roh
        self.regularisation_type = 'kldivergence'
        
    @staticmethod
    def accumulateRegularisedLoss(self, y_activation):
        roh_hat = np.mean(y_activation, axis=0)
        self.loss.reg_loss = self.loss.reg_loss + self.loss.gamma*np.sum(self.roh*np.log(self.roh/(roh_hat)) +(1-self.roh)*np.log((1-self.roh)/(1-roh_hat)))

    @staticmethod
    def regularisedLossGradient(self, y_activation):
        if self.regularisation_type == 'kldivergence':
            roh_hat = np.sum(y_activation)/y_activation.shape[-1]
            return  self.loss.gamma*(-1*self.roh/(roh_hat)) + ((1-self.roh)/(1-roh_hat))

    def forwardPass(self, inputs, outputs):
        layer_out = inputs
        for _layer in self.layers_list:
            layer_out = _layer(inputs)
            inputs = layer_out
            SparseAutoEncoder.accumulateRegularisedLoss(self, _layer.y_activation)
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
                upstream_gradient =  np.matmul(upstream_gradient, self.layers_list[len(self.layers_list) -index].w.T) + SparseAutoEncoder.regularisedLossGradient(self, _layer.y_activation)
                upstream_gradient = np.multiply(upstream_gradient,_layer.activation_derivative)
                if (len(self.layers_list)-index-1) != 0:
                    upstream_gradient_w = np.matmul(self.layers_list[len(self.layers_list) -index -2].y_activation.T,upstream_gradient)
                else:
                    upstream_gradient_w = np.matmul(inputs.T,upstream_gradient)
            upstream_gradient_b = np.sum(upstream_gradient,axis=0).T
            self.optimiser(_layer, upstream_gradient_w, upstream_gradient_b)
    
        for _layer_ in self.layers_list:
            _layer_.w = _layer_.w + _layer_.w_delta 
            _layer_.b = _layer_.b + _layer_.b_delta
