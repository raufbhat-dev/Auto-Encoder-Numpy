import numpy as np
import importlib

from autoencodernumpy import autoencoder
AutoEncoder = importlib.reload(autoencoder).AutoEncoder

class DenoisingAutoEncoder(AutoEncoder):
    def __init(self,loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch,beta):
        super().__init__(self,loss_func, optimiser_method, learning_rate, epoch, partition, mode, network_arch,beta)
        self.percentage = 0.2
            
    def addGaussianNoise(self, X, percentage):
        self.percentage = percentage
        np.random.shuffle(X)
        noise = np.random.normal(0, 1, X.shape[-1])
        non_noise_indices = np.random.randint(0, X.shape[-1], size = int(noise.size *(1 - percentage)))
        noise[non_noise_indices] = 0
        X = X + noise
        return X
    
    def addBinaryNoise(self, X, percentage):
        self.percentage = percentage
        np.random.shuffle(X)
        noise = np.random.randint(0, 1, X.shape[-1])
        non_noise_indices = np.random.randint(0, X.shape[-1], size = int(noise.size *(1 - percentage)))
        noise[non_noise_indices] = 0
        X = X + noise
        return X        
