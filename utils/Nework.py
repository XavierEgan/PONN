import numpy as np
from .Activation_Function import Activation_Function

class Network:
    def __init__(self, network_shape:list[list[Activation_Function, int]]):
        self.weights:list[np.matrix] = []
        self.biases:list[np.matrix] = []
        self.network_shape:list[list[Activation_Function, int]] = network_shape
    
    def randomize_params(self, min=-1, max=1):
        rand = np.random.default_rng()
        for i in range(1, len(self.network_shape)):
            self.weights.append(
                np.matrix(
                    [[rand.uniform(min, max) for x in range(self.network_shape[i][1])] for y in range(self.network_shape[i-1][1])]
                )
            )
            self.biases.append(
                np.matrix(
                    [rand.uniform(min, max) for x in range(self.network_shape[i][1])]
                )
            )
    
    def forward(self, input:np.matrix) -> np.matrix:
        prev_layer:np.matrix = input * self.weights[0] + self.biases[0]

        for i in range(1, len(self.weights)):
            act_func = self.activation_function(self.network_shape[i][0])
            prev_layer = act_func(prev_layer * self.weights[i] + self.biases[i])
        
        return prev_layer
    
    def activation_function(self, func:Activation_Function) -> callable:
        match func:
            case Activation_Function.InputLayer:
                raise TypeError("input layer should not input here")
            case Activation_Function.ReLU:
                def relu(x):
                    return np.max((0, x))
                vec_func = np.vectorize(relu)
                return vec_func
            case Activation_Function.Leaky_ReLU:
                def leaky_relu(x):
                    return np.max((0, x))
                vec_func = np.vectorize(leaky_relu)
                return vec_func
    
    

if __name__ == "__main__":
    network = Network([
        [Activation_Function.InputLayer, 2],
        [Activation_Function.ReLU, 3],
        [Activation_Function.ReLU, 2]
    ])
    network.randomize_layers()
    print(network.forward(np.matrix([1,1])))