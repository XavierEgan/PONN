import { Network } from "../Neural_Network/Network.ts";
import { Layer } from "../Neural_Network/Layer.ts";
import { ActivationFunction } from "../Neural_Network/Activation_Functions.ts";
import * as mathjs from 'npm:mathjs';

// test to see if the newtork forwards correctly, will test it against a hand-calculated dataset 
function network_forward_test(): boolean {
    // case 1: 2-2-2 network
    let test_network: Network = new Network(
        [
            [2, ActivationFunction.INPUT_LAYER], 
            [2, ActivationFunction.RELU],
            [2, ActivationFunction.RELU]
        ],
        [
            new Layer(
                mathjs.matrix(
                    [
                        [1, 2],
                        [1, 2]
                    ]
                ),
                mathjs.matrix(
                    [
                        [1],
                        [1]
                    ]
                ),
                ActivationFunction.RELU
            ),
            new Layer(
                mathjs.matrix(
                    [
                        [1, 2],
                        [1, 2]
                    ]
                ),
                mathjs.matrix(
                    [
                        [1],
                        [1]
                    ]
                ),
                ActivationFunction.RELU
            )
        ]
    )

    let result = test_network.forward(
        [
            [1],
            [2]
        ]
    )
    // output should be [[19],[19]]
    if (JSON.stringify(result) !== JSON.stringify([[19],[19]])) {
        return false // test has failed
    }

    result = test_network.forward(
        [
            [10],
            [9]
        ]
    )
    // output should be [[88],[88]]
    if (JSON.stringify(result) !== JSON.stringify([[88], [88]])) {
        return false // test has failed
    }
    return true
}

function layer_forward_test(): boolean {
    const test_layer = new Layer(mathjs.matrix([[1,7],[3,4],[2,2]]), mathjs.matrix([[1],[2],[3]]), ActivationFunction.LINEAR)
    const result = test_layer.forward(mathjs.matrix([[19], [19]]))
    if (JSON.stringify(result) !== JSON.stringify(mathjs.matrix([[153], [135], [79]]))) {
        return false
    }
    return true
}

function run_tests(): void {
    const tests: Array<CallableFunction> = [network_forward_test, layer_forward_test]
    tests.forEach((element, i) => {
        console.log(`test ${i}: ${element() ? "passed" : "failed"}`)
    });
}

run_tests()