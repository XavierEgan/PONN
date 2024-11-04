// made by Xavier Egan, anyone can use it as long as you follow th LICENSE file
// date this line was written: 25/20/2024 (yes i should be studying for my exams rn)
console.log("test")
import { Network } from '../Neural_Network/Network.ts';
import { ActivationFunction } from '../Neural_Network/Activation_Functions.ts';
import * as mathjs from 'npm:mathjs';

export class GeneticAlgorithm {
    generation_size: number;
    generation: Array<Network>;

    constructor(generation_size: number, layer_mask: Array<[number,number]>, starting_network: Network | null = null) {
        // initialize variables
        this.generation_size = generation_size;
        this.generation = new Array<Network>();

        // make the generation
        if (starting_network === null) {
            for (let i = 0; i < generation_size; i++) {
                this.generation.push(new Network(layer_mask));
            }
        } else {
            throw new Error("Not implemented yet");
        }
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

// try get the ai to model the function y=2x+1. It has slight variations because why not
const dataset = [
        { "x": 0, "y": 0.974695 },
        { "x": 1, "y": 3.063028 },
        { "x": 2, "y": 4.918407 },
        { "x": 3, "y": 6.976518 },
        { "x": 4, "y": 8.926179 },
        { "x": 5, "y": 10.970041 },
        { "x": 6, "y": 13.036004 },
        { "x": 7, "y": 15.035738 },
        { "x": 8, "y": 17.083317 },
        { "x": 9, "y": 19.025677 },
        { "x": 10, "y": 20.934050 },
        { "x": 11, "y": 22.933162 },
        { "x": 12, "y": 25.074768 },
        { "x": 13, "y": 26.999163 },
        { "x": 14, "y": 28.796604 },
        { "x": 15, "y": 31.099247 },
        { "x": 16, "y": 32.943290 },
        { "x": 17, "y": 35.005727 },
        { "x": 18, "y": 36.859264 },
        { "x": 19, "y": 38.910866 },
        { "x": 20, "y": 40.863343 },
        { "x": 21, "y": 42.901117 },
        { "x": 22, "y": 45.103232 },
        { "x": 23, "y": 47.122341 },
        { "x": 24, "y": 48.986191 },
        { "x": 25, "y": 50.989892 },
        { "x": 26, "y": 53.120206 },
        { "x": 27, "y": 54.983054 },
        { "x": 28, "y": 57.082224 },
        { "x": 29, "y": 59.020916 },
        { "x": 30, "y": 61.129934 },
        { "x": 31, "y": 62.953921 },
        { "x": 32, "y": 64.965455 },
        { "x": 33, "y": 67.066308 },
        { "x": 34, "y": 69.160175 },
        { "x": 35, "y": 71.035276 },
        { "x": 36, "y": 73.138344 },
        { "x": 37, "y": 75.099628 },
        { "x": 38, "y": 76.784789 },
        { "x": 39, "y": 79.077627 },
        { "x": 40, "y": 80.879648 },
        { "x": 41, "y": 83.052507 },
        { "x": 42, "y": 85.026616 },
        { "x": 43, "y": 86.962226 },
        { "x": 44, "y": 88.983254 },
        { "x": 45, "y": 90.966142 },
        { "x": 46, "y": 93.019804 },
        { "x": 47, "y": 94.794715 },
        { "x": 48, "y": 97.156448 },
        { "x": 49, "y": 99.045027 }
];

function test() {
    const start = performance.now();
    const runs = 200;
    const generation_size = 100;
    let learning_rate = 0.9;
    const learning_decay_rate = 0.995;
    const mutation_count = 3;
    const prob_of_mutation = 0.8;
    const survivor_percent = 0.1;

    const gen = new GeneticAlgorithm(
        generation_size,
        [
            [1, ActivationFunction.INPUT_LAYER],
            [2, ActivationFunction.RELU],
            [1, ActivationFunction.LINEAR],  // Simplified to direct linear mapping
        ]
    );

    // Track best fitness over time
    let best_overall_fitness = -Infinity;
    let generations_without_improvement = 0;

    for (let i = 0; i < runs; i++) {
        const fitness: Array<number> = Array.from({ length: generation_size }, () => 0);

        // Calculate fitness for each network
        for (let j = 0; j < dataset.length; j++) {
            const inputs: Array<Array<Array<number>>> = Array.from(
                { length: generation_size },
                () => [[dataset[j].x]] // double wraped becasue inputs are colum vectors, so it could be [[1],[2] etc], but since its just 1 input it looks weird
            );
            const evals = gen.get_network_evals(inputs);

            // Calculate fitness using mean squared error instead of absolute difference
            for (let k = 0; k < generation_size; k++) {
                const error = dataset[j].y - evals[k][0][0]; // [network][node][just 0 becasue its a colum vector]
                fitness[k] -= error * error;
            }
        }

        // Push and sort fitness
        gen.push_fitness(fitness);
        gen.sort_networks();

        // Track improvement
        const current_best = Math.max(...fitness);
        if (current_best > best_overall_fitness) {
            best_overall_fitness = current_best;
            generations_without_improvement = 0;
        } else {
            generations_without_improvement++;
        }

        // Run genetic algorithm
        gen.do_genetic_algorithm(learning_rate, survivor_percent, mutation_count, prob_of_mutation);
        
        // log progress
        console.log(`Generation ${i}\n -- Best Fitness = ${mathjs.max(...fitness)}\n -- Overall Best = ${best_overall_fitness}\n -- Generations Without Improvement = ${generations_without_improvement}\n -- Learning Rate = ${learning_rate}`)

        // Decay Learning Rate
        learning_rate *= learning_decay_rate
    }
    const end = performance.now();
    console.log(`Time taken: ${end - start}ms`);

    // test the best network
    const best_network = gen.get_best_network();  // Assuming gen has a method to retrieve the best network

    let test_fitness = 0;
    for (let i = 0; i < dataset.length; i++) {
        const input = [[dataset[i].x]];
        const prediction = best_network.forward(input);
        const error = dataset[i].y - prediction[0][0];
        test_fitness -= error * error;
        console.log(`input = ${dataset[i].x} | expected = ${dataset[i].y} | actual = ${prediction}`)
    }

    console.log(`Final Best Network Test Fitness: ${test_fitness}`);
}

test();

// this is in case i need it later
/*
// Test best network
        if ((i + 1) % 10 === 0) {
            const best_network = gen.get_best_network();
            console.log(`\nGeneration ${i + 1} - Testing best network:`);
            best_network.log_weights_and_biasis();
            dataset.forEach(point => {
                const pred = best_network.forward(mathjs.matrix([[point.x]]));
                console.log(`x: ${point.x}, Expected: ${point.y}, Got: ${pred[0]}`);
            });
        }
*/