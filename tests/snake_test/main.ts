import { getMatrixDataTypeDependencies } from "mathjs";
import { GeneticAlgorithm } from "../../GeneticAlgorithm/GeneticAlgorithm.ts";
import { ActivationFunction } from "../../Neural_Network/Activation_Functions.ts";

enum Direction {
    Up,
    Down,
    Left,
    Right
}

interface Coordinate {
    x: number;
    y: number;
}

class SnakeGame {
    private boxSize: number;
    private snake: Array<Coordinate> = [];
    private direction: Direction = Direction.Right;
    private expanding: number = 0;
    private food: Coordinate = { x: 0, y: 0 };
    private boardWidth: number = 20;
    private boardHeight: number = 15;
    private gameOver: boolean = false;

    constructor(boxSize: number) {
        this.boxSize = boxSize;
        this.reset();
    }

    public reset(): void {
        this.snake = [
            { x: 5, y: 3 },
            { x: 4, y: 3 },
            { x: 3, y: 3 },
            { x: 2, y: 3 },
            { x: 1, y: 3 },
        ];
        this.direction = Direction.Right;
        this.gameOver = false;
        this.spawnFood();
    }

    private spawnFood(): void {
        let newFood: Coordinate;
        do {
            newFood = {
                x: Math.floor(Math.random() * this.boardWidth),
                y: Math.floor(Math.random() * this.boardHeight)
            };
        } while (this.isCoordinateInSnake(newFood));

        this.food = newFood;
    }

    private isCoordinateInSnake(coord: Coordinate): boolean {
        return this.snake.some(segment =>
            segment.x === coord.x && segment.y === coord.y
        );
    }

    public setDirection(newDirection: Direction): void {
        // Prevent 180-degree turns
        const invalidMove =
            (this.direction === Direction.Up && newDirection === Direction.Down) ||
            (this.direction === Direction.Down && newDirection === Direction.Up) ||
            (this.direction === Direction.Left && newDirection === Direction.Right) ||
            (this.direction === Direction.Right && newDirection === Direction.Left);

        if (!invalidMove) {
            this.direction = newDirection;
        }
    }

    public physicsStep(): void {
        if (this.gameOver) return;

        // Calculate new head position
        const head = this.snake[0];
        const newHead = { x: head.x, y: head.y };

        switch (this.direction) {
            case Direction.Up:
                newHead.y -= 1;
                break;
            case Direction.Down:
                newHead.y += 1;
                break;
            case Direction.Left:
                newHead.x -= 1;
                break;
            case Direction.Right:
                newHead.x += 1;
                break;
        }

        // Check wall collision
        if (
            newHead.x < 0 ||
            newHead.x >= this.boardWidth ||
            newHead.y < 0 ||
            newHead.y >= this.boardHeight
        ) {
            this.lose();
            return;
        }

        // Check self collision
        if (this.isCoordinateInSnake(newHead)) {
            this.lose();
            return;
        }

        // Move snake
        this.snake.unshift(newHead);

        // Check food collision
        if (newHead.x === this.food.x && newHead.y === this.food.y) {
            this.expanding += 3;
            this.spawnFood();
        }

        if (this.expanding > 0) {
            this.expanding--;
        } else {
            this.snake.pop();
        }
    }

    public render(): void {
        const board = Array.from(
            { length: this.boardHeight },
            () => Array(this.boardWidth).fill(".")
        );

        // Draw snake body
        this.snake.forEach((segment, index) => {
            if (segment.y >= 0 && segment.y < this.boardHeight &&
                segment.x >= 0 && segment.x < this.boardWidth) {
                board[segment.y][segment.x] = index === 0 ? ">" : "#";
            }
        });

        // Draw food
        board[this.food.y][this.food.x] = "*";

        // Print board
        console.clear(); // Clear previous frame
        board.forEach(row => {
            console.log(row.join(""));
        });
    }

    private lose(): void {
        this.gameOver = true;
    }

    public isGameOver(): boolean {
        return this.gameOver;
    }

    public getInputs(): Array<Array<number>> {
        // 1 2 3
        // 4 H 5
        // 6 7 8
        // H = snake head
        const head = this.snake[0]
        const coordinates: Array<Coordinate> = [
            { x: head.x - 1, y: head.y + 1 },
            { x: head.x, y: head.y + 1 },
            { x: head.x + 1, y: head.y + 1 },
            { x: head.x - 1, y: head.y },
            { x: head.x + 1, y: head.y },
            { x: head.x - 1, y: head.y - 1 },
            { x: head.x, y: head.y - 1 },
            { x: head.x + 1, y: head.y - 1 },
        ]

        return coordinates.map((coord) => {
            if (coord.x == this.food.x && coord.y == this.food.y) {
                return [1];
            }
            else if (coord.y >= 0 && coord.y < this.boardHeight &&
                coord.x >= 0 && coord.x < this.boardWidth) {
                return [-1];
            }
            else if (this.isCoordinateInSnake(coord)) {
                return [-.5];
            }
            return [0];
        })
    }
}

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const number_of_runs = 1000;
const generation_size = 25;

const gen = new GeneticAlgorithm(generation_size)
gen.randomise_networks([
    [8, ActivationFunction.INPUT_LAYER],
    [8, ActivationFunction.RELU],
    [4, ActivationFunction.LINEAR]
],
-1, 1
);

const games: Array<SnakeGame> = Array(generation_size).fill(null).map(() => new SnakeGame(1));

for (let i = 0; i < number_of_runs; i++) {
    while (true) {
        const inputs: number[][][] = [];
        
        games.forEach((game) => {
            inputs.push(game.getInputs())
        });

        // get evals from nets
        const evals = gen.get_network_evals(inputs);

        // apply the evals
        games.forEach((game, i) => {
            let direction: Direction
            let largest: number = -D

        });
    }
}