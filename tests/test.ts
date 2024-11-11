import { Network } from "../Neural_Network/Network.ts";
import { ActivationFunction } from "../Neural_Network/Activation_Functions.ts";

let test_network = new Network();

test_network.load_network_from_file("test.json")

console.log(test_network.forward([[789534]])[0]);