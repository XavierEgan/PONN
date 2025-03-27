from enum import Enum

class Loss_Function(Enum):
    MEAN_SQUARED_ERROR = 1
    CROSS_ENTROPY_LOSS = 2
    BINARY_CROSS_ENTROPY = 3