import numpy as np
from ..utils.Nework import Network
from ..utils.Activation_Function import Activation_Function

class Genetic_Trainer:
    def __init__(self, generation_size:int, network_shape:list[list[int, ]], min:float, max:float):
        self.models:list[Network]

        for i in range(generation_size):
            self.models.append(
                {
                    "model" : Network(network_shape),
                    "fitness" : 0
                }
            )
            self.models[i]["model"].randomize_layers(min, max)
        
        this_gen_data = []
        overall_data = []
    
    def get_model_evals(self, inputs:list[np.matrix]) -> list[np.matrix]:
        if len(inputs) != len(self.models):
            raise ValueError("inputs are not the right size")
        
        evals:list[np.matrix] = []

        for i in range(len(inputs)):
            evals.append(self.models[i]["model"].forward(inputs[i]))
        
        return evals
    
    def put_fitness(self, fitness:list[float], track_metrics = True) -> None | dict:
        if len(fitness) != len(self.models):
            raise ValueError("fitness is not the right size")

        for i in range(len(fitness)):
            self.models[i]["fitness"] = fitness[i]

        new_model_list = [{self.models[x]["fitness"] : {"model" : self.models[x]["model"], "index" : }} for x in range(len(self.models))]


            
        


        