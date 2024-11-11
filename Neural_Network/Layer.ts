// made by Xavier Egan, anyone can use it as long as you follow th LICENSE file
// date this line was written: 25/20/2024 (yes i should be studying for my exams rn)

// the file should have the following features
// 1. neural network implimentation with the following attrubutes
//  1.1 weights
//  1.2 biases
//  1.3 activation functions (relu, leaky relu, sigmoid, tanh etc)

import * as mathjs from 'npm:mathjs';
import { ActivationFunction } from "./Activation_Functions.ts";

export class Layer {
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
    
    weights: mathjs.Matrix;
    biases: mathjs.Matrix;
    which_activation_function: number;
    activation_function: (value: number) => number;

    constructor(weights: mathjs.Matrix, biases: mathjs.Matrix, which_activation_function: number) {
        // check if weights and biases are valid, valid if weights collums == biases collums
        if (weights.size()[0] != biases.size()[0]) {
            throw new Error("");
        }
        // set variables
        this.weights = weights;
        this.biases = biases;
        
        // make sure biases and weights are correct (should be collumn matrix)
        if (biases.size()[1] > 1) {
            //the collumns are more than 1 (problem)
            throw new Error("biases are the wrong")
            
        }
        
        // set the activation function
        this.which_activation_function = which_activation_function;
        this.activation_function = this.set_activation_function(which_activation_function);
    }

    forward(input: mathjs.Matrix): mathjs.Matrix {
        // make sure the input is correct, correct if weights[1] (collum) == input[0] (row)
        if (this.weights.size()[1] != input.size()[0]) {
            throw new Error("weights collums and input rows must be the same");
        }

        // calculation is weights * input + biases then everything is run through the function
        let result: mathjs.Matrix = mathjs.multiply(this.weights, input);

        result = mathjs.add(result, this.biases);

        // apply the activation function
        result = result.map(this.activation_function);
        
        // return the result
        return result;
    }

    set_activation_function(which_activation_function: number) {
        // set the activation function to a function
        // act_func = (value) => value is shorthand for making a new funciton
        if (which_activation_function == ActivationFunction.INPUT_LAYER) {
            return () => {
                throw new Error("Input layer activation function should never be called.");
                // deno-lint-ignore no-unreachable
                return 0;
            };
        }
        else if (which_activation_function == ActivationFunction.LINEAR) {
            return (value: number) => value;
        }
        else if (which_activation_function == ActivationFunction.RELU) {
            return (value: number) => mathjs.max(value, 0);
        }
        else if (which_activation_function == ActivationFunction.LEAKY_RELU) {
            return (value: number) => mathjs.max(value, 0.01 * value);
        }
        else if (which_activation_function == ActivationFunction.SIGMOID) {
            return (value: number) => mathjs.divide(1, mathjs.add(1, mathjs.exp(-value)));
        }
        else if (which_activation_function == ActivationFunction.TANH) {
            return (value: number) => mathjs.tanh(value);
        }
        else {
            throw new Error(`activation function ${which_activation_function} is not implemented or does not exist`);
        }
    }

    clone(): Layer {
        const cloned_weights = mathjs.matrix(mathjs.clone(this.weights.valueOf()));
        const cloned_biases = mathjs.matrix(mathjs.clone(this.biases.valueOf()));
        return new Layer(cloned_weights, cloned_biases, mathjs.clone(this.which_activation_function));
    }
}

//const layerobj = new Layer(mathjs.matrix([[1,1],[2,2],[3,3]]), mathjs.matrix([[1],[2],[3]]), ActivationFunction.RELU);
//console.log(layerobj.forward(mathjs.matrix([[1],[2]])));