import numpy as np
from ..utils.Network import Network
from ..utils.Activation_Function import Activation_Function
import time
import copy

class Genetic_Trainer:
    def __init__(self, generation_size:int, network_shape:list[list[int, Activation_Function]], min:float = -1, max:float = 1):
        self.models:list[dict[str, Network | float]] = []
        self.network_shape:list[list[int, Activation_Function]] = network_shape

        for i in range(generation_size):
            self.models.append(
                {
                    "model" : Network(network_shape),
                    "fitness" : 0
                }
            )
            self.models[i]["model"].randomize_params(min, max)
        
        self.this_gen_data = {}
        self.initialize_data_tracking()

        self.rand:np.random = np.random.default_rng()

    def initialize_data_tracking(self):
        self.this_gen_data = {
            "performance":{
                "best" : 0,
                "mean" : 0
            },
            "time" : {
                "overall time" : 0,
                "get evals time" : 0,
                "put fitness time" : 0,
                "mutate time" : 0
            }
        }
    
    def get_model_evals(self, inputs:list[np.matrix]) -> list[np.matrix]:
        start_time = time.perf_counter()

        if len(inputs) != len(self.models):
            raise ValueError("inputs are not the right size")
        
        evals:list[np.matrix] = []

        for i in range(len(inputs)):
            model = self.models[i]["model"]

            evals.append(model.forward(inputs[i]))
        
        self.this_gen_data["time"]["get evals time"] += time.perf_counter() - start_time

        return evals
    
    def put_fitness(self, fitness:list[float], learning_rate:float = .01, scale:float = 1, survivor_proportion = .5, track_metrics = True) -> None | dict:
        start_time = time.perf_counter()

        if len(fitness) != len(self.models):
            raise ValueError("fitness is not the right size")

        for i in range(len(fitness)):
            self.models[i]["fitness"] = fitness[i]

        self.models = sorted(self.models, key=lambda x: x["fitness"], reverse=True)
        
        self.this_gen_data["performance"]["best"] = self.models[0]["fitness"]
        self.this_gen_data["performance"]["mean"] = sum(x["fitness"] for x in self.models) / len(self.models)

        models_keep = int(len(self.models) * survivor_proportion)
        models_remove = len(self.models) - models_keep

        self.models = self.models[0:models_keep]

        for i in range(models_remove):
            self.models.append({"model" : self.mutate(self.models[i % models_keep]["model"].clone(), learning_rate, scale), "fitness":0})
        
        self.this_gen_data["time"]["put fitness time"] += time.perf_counter() - start_time
        self.this_gen_data["time"]["overall time"] = self.this_gen_data["time"]["put fitness time"] + self.this_gen_data["time"]["get evals time"] + self.this_gen_data["time"]["overall time"]
        data = copy.deepcopy(self.this_gen_data)
        self.initialize_data_tracking()

        return data
        
    def mutate(self, model:Network, learning_rate:float, scale:float) -> Network:
        start_time = time.perf_counter()

        num_weights = sum([self.network_shape[x][0] * self.network_shape[x+1][0] for x in range(len(self.network_shape)-1)])
        num_biases = sum([x[0] for x in self.network_shape])
        num_params = num_weights + num_biases

        for layer in range(len(model.weights)):
            for weight_x in range(model.weights[layer].shape[0]):
                for weight_y in range(model.weights[layer].shape[1]):
                    if self.rand.random() < learning_rate:
                        model.weights[layer][weight_x, weight_y] += ((2 * self.rand.random()) - 1) * scale
        
        self.this_gen_data["time"]["mutate time"] += time.perf_counter() - start_time
        
        return model