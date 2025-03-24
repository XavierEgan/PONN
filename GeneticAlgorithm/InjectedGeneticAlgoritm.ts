// made by Xavier Egan, anyone can use it as long as you follow the LICENSE file
// date this line was written: 1/02/2025 (yes im lazy and havent worked on it in a month)

// this is the genetic algorithm but it makes extra "protected" networks which are random and train in a bubble and if they are still bad after a given number of generations then they are removed and replaced with more random networks. This makes it more likely for the generation to converge. Also once the secluded networked beat the main networks then they get added to the main generation and can take over.

import { Network } from '../Neural_Network/Network.ts';
import * as mathjs from 'npm:mathjs';

export class InjectedGeneticAlgorithm {
    generation_size: number;
    secluded_generation_size: number;
    secluded_generations: number;
    
    generation: Array<Network>;
    secluded_generation: Array<Network>;

    constructor(generation_size: number, secluded_generation_size: number, secluded_generations: number) {
        this.generation_size = generation_size;
        this.secluded_generation_size = secluded_generation_size;
        this.secluded_generations = secluded_generations;

        this.generation = new Array<Network>;
        this.secluded_generation = new Array<Network>;
    }

    randomize_networks(layer_mask: Array<[number,number]>, min: number, max: number) {
        // randomize the main generation
        for (let i=0; i<this.generation_size; i++) {
            const network = new Network()
            network.randomise_weights(layer_mask, min, max)
            this.generation.push(network)
        }

        
    }
    
    randomize_secluded_networks(layer_mask)
}