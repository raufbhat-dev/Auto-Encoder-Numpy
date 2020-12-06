import os
import struct
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Utility:
    def __init__(self):
        pass
    
    @staticmethod
    def dataScale(X):
        mean = np.nanmean(X, axis = 0)
        std = np.nanstd(X, axis = 0)
        X = X  - mean
        std[std == 0.0] = 1.0
        X = X/std
        return X

    @staticmethod
    def readIDX(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    @staticmethod
    def closestDivisors(n):
        a = round(math.sqrt(n))
        while n%a > 0: a -= 1
        return a,n//a
        
    @staticmethod
    def plotWeights(layers, save = False, **kwargs):
        fig = plt.figure(figsize=(10, 10))
        for index, _layer in enumerate(layers):
            rows, columns = Utility.closestDivisors(_layer.w.shape[-1])
            print('layer: {}'.format(index))
            image_shape = Utility.closestDivisors(_layer.w.shape[0])
            for i in range(columns*rows):
                layer_temp = _layer.w.T[i].reshape(image_shape)
                ax = fig.add_subplot(rows, columns, i+1)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                plt.imshow(layer_temp, interpolation='bilinear', cmap=plt.cm.Greys_r)
            plt.show()
            if save:
                os.makedirs(os.path.dirname(os.path.join(BASE_DIR, 'Outputs/')), exist_ok = True)
                plt.savefig(os.path.join(BASE_DIR, 'Outputs/{}_Layer{}_Weights_{}.png'.format(_layer['network_part'],index)))
                print('Outputs Saved: {}'.format(BASE_DIR))
    