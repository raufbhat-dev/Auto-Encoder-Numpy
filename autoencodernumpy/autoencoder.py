import importlib

from neuralnetnumpy import layer, neuralnet

NeuralNet = importlib.reload(neuralnet).NeuralNet
layer = importlib.reload(layer)

class AutoEncoder(NeuralNet):
    def __init__(self, loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch,**kwargs):
        self.encoder_layer_list = [] 
        self.decoder_layer_list = []
        super().__init__(loss_func, optimiser_method, learning_rate, epoch, partition_size, mode, network_arch,**kwargs)
        
                    
    def createNetwork(self):
        network_layers = []
        for index, _layer in  enumerate(self.network_arch):
                if _layer['layer_type'].lower() != 'input': 
                    if _layer['network_part'].lower() == 'encoder':
                        self.encoder_layer_list.append(layer.Layer(self.network_arch[index-1]['size'],_layer['size'], _layer['activation'],_layer['layer_type']))            
                        self.layers_list.extend(self.encoder_layer_list)
                    elif _layer['network_part'].lower() == 'decoder':
                        self.decoder_layer_list.append(layer.Layer(self.network_arch[index-1]['size'],_layer['size'], _layer['activation'],_layer['layer_type']))                               
                        self.layers_list.extend(self.decoder_layer_list)
