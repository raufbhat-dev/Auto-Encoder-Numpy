class GradientDescent:
    def __init__(self, momentum = False, learning_rate = 0.01, **kwargs):
        if momentum:
            self.optimiser_func = 'momentum'
            self.beta = kwargs['beta']
        else:
            self.optimiser_func = 'sgd'
        self.learning_rate = learning_rate
    
    def __call__(self, layer, upstream_gradient_w, upstream_gradient_b):
        if self.optimiser_func.lower() == 'momentum':
            layer.w_delta =  self.beta*layer.w_delta -1*self.learning_rate*upstream_gradient_w
            layer.b_delta = self.beta*layer.b_delta -1*self.learning_rate*upstream_gradient_b
        elif self.optimiser_func.lower() == 'sgd':
            layer.w_delta = -1*self.learning_rate*upstream_gradient_w 
            layer.b_delta = -1*self.learning_rate*upstream_gradient_b
