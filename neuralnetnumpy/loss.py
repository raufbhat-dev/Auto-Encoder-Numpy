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
                self.loss = -1*(np.sum(np.multiply(y,np.log(y_pred)) + np.multiply((1-y),np.log(1- y_pred))))/y.shape[-1]
                self.loss_derivative = -1*(np.divide(y,y_pred)+ np.divide((1-y),(1 - y_pred))) 
        
        return func


class RegularisedLoss(Loss):
    def __init__(self, loss_func, regularisation_type, gamma):
        super().__init__(loss_func)
        self.regularisation_type = regularisation_type
        self.gamma = gamma
        self.reg_loss = 0.0
        self.reg_derivative = np.zeros(1)
        self.loss_total = self.loss + self.reg_loss
    
    def accumulateRegularisedLoss(self, parameter):
        if self.regularisation_type == 'l2':
            self.reg_loss = self.reg_loss + self.gamma*np.trace(parameter*parameter.T)

        if self.regularisation_type == 'l1':
            self.reg_loss = self.reg_loss + self.gamma*np.trace(parameter)
        
    def regularisedLossGradient(self, parameter):
        if self.regularisation_type == 'l2':
            self.reg_derivative = (self.gamma/parameter.shape[-1])*parameter

        if self.regularisation_type == 'l1':
            self.reg_derivative = (self.gamma/parameter.shape[-1])*np.ones(parameter.shape)

        if self.regularisation_type == None:
            self.reg_derivative = np.zeros(parameter.shape)
        
        return self.reg_derivative

    def getLossTotal(self):
        self.loss_total = self.loss + self.reg_loss
        self.reg_loss = 0
