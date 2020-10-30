import numpy as np

class Loss:
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.loss_derivative =  np.zeros(1)
        self.loss =  0.0
    
    def getLoss(self):
        if self.loss_func.lower() == 'meansquared':
            def func(y_pred,y):
                error =  y - y_pred
                self.loss = np.sum(np.diag(np.matmul(error,error.T))) / y.shape[-1]
                self.loss_derivative = -1*error
        if self.loss_func.lower() == 'crossentropy':
            def func(y_pred,y):
                epsilon = 1e-12 
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                self.loss = np.sum(-1*(np.multiply(y,np.log(y_pred))+np.multiply((1-y),np.log(1- y_pred))))/y_pred.shape[-1]
                self.loss_derivative = -1*(np.divide(y, y_pred))+ np.divide(1-y ,1 - y_pred) 
        return func
