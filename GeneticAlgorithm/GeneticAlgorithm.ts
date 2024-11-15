// made by Xavier Egan, anyone can use it as long as you follow th LICENSE file
// date this line was written: 25/20/2024 (yes i should be studying for my exams rn)
console.log("test")
import { Network } from '../Neural_Network/Network.ts';
import * as mathjs from 'npm:mathjs';

export class GeneticAlgorithm {
    generation_size: number;
    generation: Array<Network>;

    constructor(generation_size: number) {
        // initialize variables
        this.generation = new Array<Network>();
        this.generation_size = generation_size;
    }

    randomise_networks(layer_mask: Array<[number, number]>, min: number, max: number) {
        /*
        initialize the generation by randomising all the networks
        */
        // randomise the networks
        for (let i = 0; i < this.generation_size; i++) {
            const network = new Network();
            network.randomise_weights(layer_mask, min, max)
            this.generation.push(network);
        }
    }

    initialize_generation_with_network_from_file(path: string) {
        /*
        initialize the generation by making clones of the network at `path`
        */
        for (let i = 0; i < this.generation_size; i++) {
            const network = new Network();
            network.load_network_from_file(path)
            this.generation.push(network);
        }
    }

    write_best_network_to_file(path: string) {
        this.get_best_network().write_network_to_file(path)
    }

    get_network_evals(inputs: Array<Array<Array<number>>>): Array<Array<Array<number>>> {
        // check the inputs
        if (inputs.length != this.generation_size) {
            throw new Error(`input length is not the same size as generation size, expected ${this.generation_size}, got ${inputs.length}`)
        }

        const evals = new Array<Array<Array<number>>>();
        // get the outputs of networks
        for (let i=0; i < this.generation.length; i++) {
            // get the output of the network
            const e: Array<Array<number>> = this.generation[i].forward(
                inputs[i]
            )
            evals.push(e);
        }
        // return the model evaluations
        return evals
    }

    push_fitness(fitness: Array<number>) {
        // set the fitness of all the networks ( run this and THEN run do_genetic_algorithm() )
        for (let i = 0; i < fitness.length; i++) {
            this.generation[i].fitness = fitness[i];
        }
    }

    sort_networks(): void{
        // sort the generation by fitness in descending order
        this.generation.sort((a, b) => b.fitness - a.fitness);
    }

    get_best_network(): Network{
        // returns the best network
        return this.generation[0]
    }

    do_genetic_algorithm(learning_rate: number = 0.5, survivor_percent: number = 0.5, num_mutations: number = 4, prob_of_mutation = 1) {
        // remove the bottom survivor_percent% of the generation
        this.generation = this.generation.splice(0, Math.floor(this.generation.length * survivor_percent));
        
        // copy the top (100-survivor_percent)% of the generation
        let i = 0;
        const clones: Array<Network> = [];
        while (clones.length + this.generation.length < this.generation_size) {
            // add the new network to
            clones.push(this.generation[i].clone())

            // set i to i+1%new_bottom_length This will cycle i between 0 and new_bottom_length-1, since when i+1 is a multiple of new_bottom_length it will become 0 (im proud of this line)
            i = (i + 1) % this.generation.length
        }

        // set the generation to the survivors and the clones
        this.generation = [...this.generation, ...clones]
        
        // mutate the networks
        for (let i=0; i < this.generation.length; i++) {
            // apply the probability of mutation
            if (mathjs.random() > prob_of_mutation) {
                continue
            }
            this.generation[i].mutate(learning_rate, num_mutations)
        }
    }
}