const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const Direction = Object.freeze({
    up: 0,
    down: 1,
    left: 2,
    right: 3
});

class Snake_Game {
    box_size: number
    constructor(box_size: number) {
        this.box_size = box_size
        this.reset()

        const canvas = document.getElementById("gameCanvas");
        const ctx = canvas.getContext("2d");
    }

    reset() {
        this.snake = [
            { x: 1 * this.box_size, y: 3 * this.box_size },
            { x: 2 * this.box_size, y: 3 * this.box_size },
            { x: 3 * this.box_size, y: 3 * this.box_size },
            { x: 4 * this.box_size, y: 3 * this.box_size },
            { x: 5 * this.box_size, y: 3 * this.box_size },
        ];
        this.direction = Direction.left;
    }

    phyics_step() {
        // get the offset the new head of the snake should be
        let offset = { x: 0, y: 0 }
        switch (this.direction) {
            case Direction.up:
                offset.y = 1
                break;
            case Direction.down:
                offset.y = -1
                break;
            case Direction.left:
                offset.x = -1
                break;
            case Direction.right:
                offset.x = 1
                break;
        }

        // apply the offset by making a new snake head

    }

    render() {
        this.snake.forEach((cell) => {
            this.canvas.fillRect(cell.x, cell.y, this.box_size, this.box_size)
        });
    }

    loose() {
        this.reset();
    }
}

game = new Snake_Game(10);

game.render()