import numpy as np
from ..utils.Nework import Network
from ..utils.Activation_Function import Activation_Function
from ..utils.Loss_Function import Loss_Function

"""
Resources used:
 - claude 3.7 (for understanding, all code was written by me (Xavier))
 - https://www.geeksforgeeks.org/adam-optimizer/
 - https://www.geeksforgeeks.org/backpropagation-in-neural-network/

"""

class Adam_Optimizer:
    def __init__(self, network:Network):
        self.network:Network = network

        # hyperparams
        self.learn_rate = .001
        self.b1 = .9
        self.b2 = .999
        self.epsilon = 10.0**-8

        self.t = 0

        self.wmt:list[np.matrix] = []
        self.wvt:list[np.matrix] = []

        self.bmt:list[np.matrix] = []
        self.bvt:list[np.matrix] = []

        for layer in self.network.weights:
            self.wmt.append(np.zeros(layer.shape))
            self.wvt.append(np.zeros(layer.shape))

        for layer in self.network.biases:
            self.bmt.append(np.zeros(layer.shape))
            self.bvt.append(np.zeros(layer.shape))

    #TODO: Fix Cross entropy

    def loss(self, loss:Loss_Function) -> callable:
        match loss:
            case Loss_Function.MEAN_SQUARED_ERROR:
                # from claude (i wrote the code myself tho)
                def mse(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    return (y - y_correct)**2
                return mse
            
        """ case Loss_Function.CROSS_ENTROPY_LOSS:
                # from claude (i wrote the code myself tho)
                def cel(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    return -(y_correct * np.log(y) + (1-y_correct) * np.log(1-y))
                return cel
            
            case Loss_Function.BINARY_CROSS_ENTROPY:
                # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/
                def bce(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    # y is the prediction from the model (probability that it is in the correct category)
                    # y_correct is 
                    return -(y_correct*np.log(y) + (1-y_correct) * np.log(1-y))
                return bce"""
    
    def loss_derivative(self, loss:Loss_Function) -> callable:
        match loss:
            case Loss_Function.MEAN_SQUARED_ERROR:
                # from claude (i wrote the code myself tho)
                def mse(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    return y - y_correct # dont add the 2 cuz its easier without
                return mse
            
        """ case Loss_Function.CROSS_ENTROPY_LOSS:
                # from claude (i wrote the code myself tho)
                def cel(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    return (y - y_correct) / (y * (1 - y) + self.epsilon)
                return cel
            
            case Loss_Function.BINARY_CROSS_ENTROPY:
                # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/
                def bce(y:np.matrix, y_correct:np.matrix) -> np.matrix:
                    # y is the prediction from the model (probability that it is in the correct category)
                    # y_correct is 
                    return (y - y_correct) / (y * (1 - y) + self.epsilon)
                return bce"""

    def train(self, input:np.matrix, expected_outputs:np.matrix, loss:Loss_Function):
        self.t +=1

        output = self.network.forward(input)
        error =  output - expected_outputs

        loss = self.loss()

        loss_gradient = self.backpropogate()
        w_loss_grad = loss_gradient[0]
        b_loss_grad = loss_gradient[1]

        wgrad = []
        bgrad = []

        # weights (sorta vectorized so it does it all at once)
        for i in range(len(self.wmt)):
            # First moment (mean) estimate
            self.wmt[i] = self.b1 * self.wmt[i] + (1 - self.b1) * w_loss_grad[i]

            # Second moment (variance) estimate
            self.wvt[i] = self.b2 * self.wvt[i] + (1-self.b2) * (w_loss_grad[i]**2)

            # Bias correction (move them away from zero)
            mht = self.wmt[i]/(1-self.b1**self.t)
            mvt = self.wvt[i]/(1-self.b2**self.t)

            self.network.weights[i] = self.network.weights[i] - (mht * self.learn_rate) / (np.sqrt(mvt) + self.epsilon)
        
        # weights (sorta vectorized so it does it all at once)
        for i in range(len(self.bmt)):
            # First moment (mean) estimate
            self.bmt[i] = self.b1 * self.bmt[i] + (1 - self.b1) * b_loss_grad[i]

            # Second moment (variance) estimate
            self.bvt[i] = self.b2 * self.bvt[i] + (1-self.b2) * (b_loss_grad[i]**2)

            # Bias correction (move them away from zero)
            mht = self.bmt[i]/(1-self.b1**self.t)
            mvt = self.bvt[i]/(1-self.b2**self.t)

            self.network.biases[i] = self.network.biases[i] - (mht * self.learn_rate) / (np.sqrt(mvt) + self.epsilon)

    def backpropogate(self, loss:Loss_Function) -> list[list[np.matrix]]:
        #TODO: impliment this
        # 