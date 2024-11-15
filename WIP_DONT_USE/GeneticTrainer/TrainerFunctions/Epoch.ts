import { Network } from "../../../Neural_Network/Network.ts";
import { GeneticAlgorithm } from "../../../GeneticAlgorithm/GeneticAlgorithm.ts";
import { ActivationFunction } from "../../../Neural_Network/Activation_Functions.ts";
import * as mathjs from "npm:mathjs"

export class Epoch {
    train(dataset: Array<[Array<Array<number>>, Array<Array<number>>]>, fitness_function: CallableFunction, runs: number, generation_size: number, learning_rate: number, learning_decay_rate: number, mutation_count: number, prob_of_mutation: number, survivor_percent: number) {
        const start = performance.now();
        const gen = new GeneticAlgorithm(generation_size);
        /*
        dataset: [[[input], [expected output]]]
        */

        gen.randomise_networks(
            [
                [1, ActivationFunction.INPUT_LAYER],
                [2, ActivationFunction.LINEAR],
                [1, ActivationFunction.LINEAR]
            ],
            -2, 2
        )

        // Track best fitness over time
        let best_overall_fitness = -Infinity;
        let generations_without_improvement = 0;

        for (let i = 0; i < runs; i++) {
            const fitness: Array<number> = Array.from({ length: generation_size }, () => 0);
            // for each dataset value
            for (let j = 0; j < dataset.length; j++) {
                const inputs: Array<Array<Array<number>>> = Array.from(
                    { length: generation_size },
                    () => dataset[j][0]
                );
                const evals = gen.get_network_evals(inputs);

                // Calculate fitness using mean squared error instead of absolute difference
                for (let k = 0; k < generation_size; k++) {
                    const error = dataset[j][1][0] - evals[k][0][0]; // [network][node][just 0 becasue its a colum vector]
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

            // dont randomise weights at the end
            if (i == runs - 1) {
                console.log("Finished Training")
                break;
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
        const best_network = gen.get_best_network();
        best_network.write_network_to_file("test.json");

        let test_fitness = 0;
        for (let i = 0; i < dataset.length; i++) {
            const input = dataset[i][0];
            const prediction = best_network.forward(input);
            let error: number
            // calculate the error
            for (let j=0; j<dataset[i][1].length; j++) {
                error -= mathjs.abs(dataset[i][0][j]-prediction)
            }
            test_fitness -= error * error;
            console.log(`input = ${input} | expected = ${dataset[i].y} | actual = ${prediction}`)
        }

        console.log(`Final Best Network Test Fitness: ${test_fitness}`);

        gen.write_best_network_to_file("test.json");
    }
}