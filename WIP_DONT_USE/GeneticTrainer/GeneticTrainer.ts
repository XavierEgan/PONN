import { Environment } from "./TrainerFunctions/Environment.ts"
import { Epoch } from "./TrainerFunctions/Epoch.ts"

export class Genetic_Trainer {
    Epochs: Epoch;
    Environment: Environment;

    constructor() {
        this.Epochs = new Epoch;
        this.Environment = new Environment;
    }
}