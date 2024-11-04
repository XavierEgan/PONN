// made by Xavier Egan, anyone can use it as long as you follow th LICENSE file
// date this line was written: 25/20/2024 (yes i should be studying for my exams rn)
import { Layer } from "./Layer.ts";
import { ActivationFunction } from "./Activation_Functions.ts";
import * as mathjs from 'npm:mathjs';
import type { Matrix } from "npm:mathjs";
import { rightArithShiftDependencies } from "npm:mathjs";
import { number } from "npm:mathjs";

export class Network {
    /*
    layer mask is like this (the activation node of the first layer does not change anything since its the input) also(the last layer is the output layer):
    [
        [num_nodes, ActivationFunction.function]
    ]
    */
    
    layer_mask: Array<[number, number]>;
    layers: Array<Layer>;

    fitness: number;

    constructor(layer_mask: Array<[number, number]>, layers: Array<Layer> | null = null) {
        // initalise fitness as 0
        this.fitness = 0;
        
        // initialise the layer mask
        this.layer_mask = layer_mask;

        // make the layers
        this.layers = new Array<Layer>();
        if (layers === null) {
            // we need to randomise the weights
            for (let i=1; i<layer_mask.length; i++) {
                // for each layer
                // Weights
                /*
                the weights are stored like this
                matrix_dimention = output_size x input_size
                | a  d |
                | b  e |
                | c  f |

                biases are stored like this
                matrix_dimention = output_size x 1
                | a |
                | b |
                | c |

                activation functions are stored like this
                "relu" or "leaky_relu" or "sigmoid" or "tanh"
                */
                const output_size = layer_mask[i][0];
                const input_size = layer_mask[i-1][0];

                // wth was i doing before this is so much easier
                const randomised_weights: mathjs.Matrix = mathjs.matrix(mathjs.random([output_size, input_size], -10, 10));
                const randomised_biases: mathjs.Matrix = mathjs.matrix(mathjs.random([output_size, 1], -10, 10));
                
                this.layers.push(new Layer(randomised_weights, randomised_biases, layer_mask[i][1]));
            }
        }
        else{
            // Initialize with provided layers
            this.layers = layers;
        }

    }

    forward(input: Array<Array<number>>): Array<Array<number>> {
        // make sure the input is the correct size
        if (input.length != this.layer_mask[0][0]) {
            throw new Error(`Input length is not the right length, expected: ${this.layer_mask[0][0]}, got: ${input.length}`);
        }

        // turn the input into a matrix
        let output = mathjs.matrix(input)

        // loop through all the layers and perform the feedforward algorithm
        for (let i = 0; i < this.layers.length; i++){
            output = this.layers[i].forward(output);
        }

        // turn output into array
        const output_array = output.toArray() as Array<Array<number>>
        if (!Array.isArray(output_array[0])) {
            throw new Error("output_array is not the corect shape")
        }
        return output_array
    }

    mutate(learning_rate: number = 0.5, num_mutations: number = 4): void {
        for (let i=0; i < num_mutations; i++) {
            // select a random layer (same for weight and bias)
            const layer: Layer = this.layers[mathjs.randomInt(0, this.layers.length)]

            // WEIGHT
            // select random weight index
            const random_weight_index = [
                mathjs.randomInt(0, layer.weights.size()[0]), 
                mathjs.randomInt(0, layer.weights.size()[1])
            ]

            // mutate the weight
            const current_weight = layer.weights.get(random_weight_index)
            const mutated_weight = current_weight + (current_weight * learning_rate * (mathjs.random() - .5))
            layer.weights.set(random_weight_index, mutated_weight)

            // BIAS
            // select random bias index
            const random_bias_index = [
                mathjs.randomInt(0, layer.biases.size()[0]),
                mathjs.randomInt(0, layer.biases.size()[1])
            ]
            
            // mutate the bias
            const current_bias = layer.biases.get(random_bias_index)
            const mutated_bias = current_bias - (current_bias * learning_rate * (mathjs.random() - .5))
            layer.biases.set(random_bias_index, mutated_bias)
        }
    }

    clone(): Network {
        const new_layers: Array<Layer> = [];
        // Create a new Network object
        for (let i = 0; i < this.layers.length; i++) {
            new_layers.push(this.layers[i].clone());
        }

        // construct and return the network
        return new Network(this.layer_mask, new_layers);
    }

    log_weights_and_biasis(): void {
        // print all the weights and biases
        for (let i=0; i < this.layers.length; i++) {
            const layer = this.layers[i]

            console.log(`\nLayer ${i+1}`)
            // Weights
            console.log(`\n - Weights: ${layer.weights.toString()}`)

            // Biases
            console.log(`\n - Biases: ${layer.biases.toString()}`)
        }
    }
}