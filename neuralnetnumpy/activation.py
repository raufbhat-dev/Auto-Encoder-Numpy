import numpy as np

class Activation:
    def __init__(self,activation):
        self.activation = activation
        self.activation_derivative = np.ones(shape = (1,1))
    
    def getActivation(self):
        if self.activation.lower() == 'sigmoid':
            def func(y):
                y_ret = np.matrix(1/(1+np.exp(1-0.01*y)*np.exp(-1)))
                self.activation_derivative = np.multiply(y_ret, np.ones(y_ret.shape[-1])- y_ret)
                return y_ret
        elif self.activation.lower() == 'relu':
            def func(y):
                y_ret = np.where(y<0, 0, y)
                self.activation_derivative = np.matrix(np.where(y>0, 1, 0))
                return y_ret
        elif self.activation.lower() == 'leakyrelu':
            def func(y):
                alpha = 0.03
                y_ret = np.where(y > 0, y, y*alpha) 
                self.activation_derivative = np.matrix(np.where(y>0, 1, alpha))
                return y_ret
        elif self.activation.lower() == 'softmax':
            def func(y):
                shift_y = y - np.max(y)
                exps = np.exp(shift_y)
                softmax = np.array(exps / np.sum(exps,axis=1))
                self.activation_derivative = np.ones(y.shape[-1])
                return softmax
        elif self.activation.lower() == 'tanh':
            def func(y):
                act_tanh = np.tanh(y)
                self.activation_derivative = (1 - np.power(act_tanh, 2))/2 
                return act_tanh
        elif self.activation.lower() == 'softplus':
            def func(y):
                y_ret = np.log(1.0 + (np.exp(y)))
                self.activation_derivative = 1.0 / (1.0 + np.exp(-y))
                return y_ret
        return func
