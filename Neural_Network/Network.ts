// made by Xavier Egan, anyone can use it as long as you follow th LICENSE file
// date this line was written: 25/20/2024 (yes i should be studying for my exams rn)
import { Layer } from "./Layer.ts";
import * as mathjs from 'npm:mathjs';

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

    constructor() {
        // initalise fitness as 0
        this.fitness = 0;

        // make the layers
        this.layers = new Array<Layer>();

        // layer mask
        this.layer_mask = new Array<[number, number]>();
    }

    randomise_weights(layer_mask: Array<[number, number]>) {
        if (layer_mask.length == 0) {
            throw new Error(`layer mask is not defined, got ${layer_mask}`)
        }
        // set the layer mask
        this.layer_mask = layer_mask

        // we need to randomise the weights
        for (let i = 1; i < this.layer_mask.length; i++) {
            // for each layer
            const output_size = this.layer_mask[i][0];
            const input_size = this.layer_mask[i - 1][0];

            // wth was i doing before this is so much easier
            const randomised_weights: mathjs.Matrix = mathjs.matrix(mathjs.random([output_size, input_size], -10, 10));
            const randomised_biases: mathjs.Matrix = mathjs.matrix(mathjs.random([output_size, 1], -10, 10));

            this.layers.push(new Layer(randomised_weights, randomised_biases, layer_mask[i][1]));
        }
    }

    load_network_from_file(path: string) {
        // read the json file
        let json_string: string
        try {
            json_string = Deno.readTextFileSync(path);
        } catch(e) {
            if (e instanceof Deno.errors.NotFound) {
                throw new Error(`File not found ${path}`)
            } else {
                throw e
            }
        }

        // load the data into the network
        this.load_network_from_json_string(json_string)
    }

    load_network_from_json_string(json_string: string) {
        // parse the data
        const data = JSON.parse(json_string);

        // make sure the data is correct
        if (!(("layer_mask" in data) && ("layers" in data))) {
            throw new Error(`JSON Data is incorrect`)
        }

        // load the layer mask in
        this.layer_mask = data.layer_mask
        
        // make the layers
        for (let i = 0; i < data.layers.length; i++) { 
            // weights
            const weights = mathjs.matrix(data.layers[i].weights)

            // biases
            const biases = mathjs.matrix(data.layers[i].biases)

            // add the layer
            this.layers.push(new Layer(weights, biases, this.layer_mask[i+1][1]))
        }
    }

    get_json_string() {
        return JSON.stringify({
            layer_mask: this.layer_mask,
            layers: this.layers.map(layer => ({
                weights: layer.weights.toArray(),
                biases: layer.biases.toArray()
            }))
        }, null, 4);
    }

    write_network_to_file(path: string) {
        Deno.writeTextFileSync(path, this.get_json_string())
    }

    forward(input: Array<Array<number>>): Array<Array<number>> {
        // make sure the input is the correct size
        if (input.length != this.layer_mask[0][0]) {
            throw new Error(`Input length is not the right length, expected: ${this.layer_mask[0][0]}, got: ${input.length}`);
        }

        // make sure the input is a matrix (just check the first element for performance)
        if (!(input[0] instanceof Array)) {
            throw new Error(`input is not a matrix`)
        }

        // turn the input into a matrix
        let output = mathjs.matrix(input);

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
        // make a new network and load the weights/biases
        const new_network = new Network();
        new_network.load_network_from_json_string(this.get_json_string());

        // return the net
        return new_network;
    }
}