import numpy as np


class Adagrad:
    def __init__(self, dim):
        self.dim = dim
        self.eps = 1e-3

        # initial learning rate
        # self.learning_rate = 0.05  #original 100-dim depnn
        # self.learning_rate = 0.0001  #crf 100-dim
        # self.learning_rate = 0.02
        self.learning_rate = 0.01
        # self.learning_rate = 0.001
        # stores sum of squared gradients 
        self.h = np.zeros(self.dim)

    def rescale_update(self, gradient):
        curr_rate = np.zeros(self.h.shape)
        self.h += gradient ** 2
        curr_rate = self.learning_rate / (np.sqrt(self.h) + self.eps)
        return curr_rate * gradient

    def reset_weights(self):
        self.h = np.zeros(self.dim)
