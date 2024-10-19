const app = new PIXI.Application({
    width: 1420,
    height: 420,
    backgroundColor: 0x000000,
    view: document.getElementById('myCanvas')
});

const gridContainer = new PIXI.Container();
app.stage.addChild(gridContainer);

// Create an array of obstacle lines
const obstacles = [
    // { x1: 20, y1: 50, x2: 20, y2: 420 },
    // { x1: 80, y1: 50, x2: 80, y2: 420 },
    // { x1: 20, y1: 420, x2: 80, y2: 420 },
    // { x1: 20, y1: 50, x2: 80, y2: 50 },
    // Add more obstacle lines as needed
];

function drawGrid() {
    const gridLines = new PIXI.Graphics();
    gridLines.lineStyle(1, 0xFFFFFF, 0.2);
    for (let x = 0; x <= 1420; x += 10) {
        gridLines.moveTo(x, 0);
        gridLines.lineTo(x, 420);
    }
    for (let y = 0; y <= 420; y += 10) {
        gridLines.moveTo(0, y);
        gridLines.lineTo(1420, y);
    }
    gridContainer.addChild(gridLines);

    gridLines.lineStyle(1, 0xFFFFFF, 1);
    for (let x=12; x < 140; x += 12){
        gridLines.moveTo(x*10, 5*10);
        gridLines.lineTo(x*10, 42*10);
        obstacles.push({x1: x*10, y1: 5*10, x2: x*10, y2: 42*10})

        gridLines.moveTo((x+8)*10, 5*10);
        gridLines.lineTo((x+8)*10, 42*10);
        obstacles.push({x1: (x+8)*10, y1: 5*10, x2: (x+8)*10, y2: 42*10})

        gridLines.moveTo(x*10, 42*10);
        gridLines.lineTo((x+8)*10, 42*10);
        obstacles.push({x1: (x)*10, y1: 42*10, x2: (x+8)*10, y2: 42*10})

        gridLines.moveTo(x*10, 5*10);
        gridLines.lineTo((x+8)*10, 5*10);
        obstacles.push({x1: (x)*10, y1: 5*10, x2: (x+8)*10, y2: 5*10})
    }

    gridLines.lineStyle(1, 0xFF0000, 1);
    const thing = [12, 20, 24, 32, 36, 44, 48, 56, 60, 68, 72, 80, 84, 92, 96, 104, 108, 116, 120, 128, 132, 140]
    for (let x=0; x<thing.length; x++){
        gridLines.moveTo(thing[x]*10, 5*10);
        gridLines.lineTo(thing[x]*10, 42*10);
    }



    gridLines.lineStyle(1, 0x00FF00, 1);
    for (let x=2; x < 8; x++){
        gridLines.moveTo(2*10, 5*10);
        gridLines.lineTo(2*10, 42*10);
        obstacles.push({x1: (2)*10, y1: 5*10, x2: (2)*10, y2: 42*10})

        gridLines.moveTo((8)*10, 5*10);
        gridLines.lineTo((8)*10, 42*10);
        obstacles.push({x1: (8)*10, y1: 5*10, x2: (8)*10, y2: 42*10})

        gridLines.moveTo(2*10, 42*10);
        gridLines.lineTo((8)*10, 42*10);
        obstacles.push({x1: (2)*10, y1: 42*10, x2: (8)*10, y2: 42*10})

        gridLines.moveTo(2*10, 5*10);
        gridLines.lineTo((8)*10, 5*10);
        obstacles.push({x1: (2)*10, y1: 5*10, x2: (8)*10, y2: 5*10})
    }

    gridLines.lineStyle(1, 0x0000FF, 1);
    for (let x=1; x < 142; x += 2){
        gridLines.moveTo(x*10, 0*10);
        gridLines.lineTo((x+1)*10, 0*10);

        gridLines.moveTo((x)*10, 1*10);
        gridLines.lineTo((x+1)*10, 1*10);
        obstacles.push({x1: (x)*10, y1: 1*10, x2: (x+1)*10, y2: 1*10})

        gridLines.moveTo(x*10, 0*10);
        gridLines.lineTo((x)*10, 1*10);

        gridLines.moveTo((x+1)*10, 0*10);
        gridLines.lineTo((x+1)*10, 1*10);
    }
}

function createCar(x, y, angle) {
    const car = new PIXI.Graphics();
    car.beginFill(0xFF0000);
    car.drawRect(-20 / 2, -10 / 2, 20, 10);
    car.endFill();
    car.x = x;
    car.y = y;
    car.rotation = angle;
    return car;
}

// Draw the grid
drawGrid();

const cars = [];
const lidarLines = [];

for (let x=0; x<10; x++){
    const car = createCar(0, 0, 90);
    app.stage.addChild(car)
    cars.push(car)
    const lidarLine = createLidarLines(car, 5, 50);
    lidarLines.push(lidarLine);
    app.stage.addChild(lidarLine);
}
// const car2 = createCar(250, 20, 180);
// app.stage.addChild(car);
// app.stage.addChild(car2);


function update(instance_num_, x, y, angle) {
    // console.log(x)
    // car.rotation += 0.01;
    cars[instance_num_].x = x*10
    cars[instance_num_].y = y*10
    cars[instance_num_].rotation = angle
    
    // Update the LIDAR lines for the specific car after it moves
    updateLidarLines(cars[instance_num_], lidarLines[instance_num_], obstacles);

    // requestAnimationFrame(update);
    // requestAnimationFrame(() => update(instance_num_, x, y));

}

const socket = io("http://localhost:8000", { transports: ['websocket'] });

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Client disconnected');
    // Clean up socket resources
    socket.close();
});

socket.on('grid_update', (data) => {
    // console.log('Received grid_update:', data);  
    updateDynamicElements(data["next_move"]);
    updateDynamicTasks(data["tasks"]);
    updateContainer(data["container"]);
    averageSpeed = data["speed"];
});

function updateContainer(data){
    containerCount = data;
}

function updateSpeed(data){
    console.log(data);
    averageSpeed = data;
}

function updateDynamicElements(data) {
    // console.log(JSON.stringify(data));
    // console.log('updateDynamicElements called with:', JSON.stringify(data));
    for (let i=0; i<data.length; i++){
        // console.log(i);
        // console.log(String(data[i][0]));
        // console.log(String(data[i][1]));
        update(i, data[i][0], data[i][1], data[i][2]);
    }
    }

function createImageSprite(x, y) {
    // Load the image texture
    // imagePath = 'images/transportation.png'
    const texture = PIXI.Texture.from(imagePathShip);

    // Create a sprite using the texture
    const imageSprite = new PIXI.Sprite(texture);

    imageSprite.width = 25;
    imageSprite.height  = 25;

    // Optionally, set the anchor point to the center (0.5 for both x and y axis)
    imageSprite.anchor.set(0.5);

    // Set the position of the sprite
    imageSprite.x = x;
    imageSprite.y = y;

    return imageSprite;
}

function createImageSpriteD(x, y) {
    // Load the image texture
    // imagePath = 'images/transportation.png'
    pic = Math.round(Math.random()*3);
    let texture
    if (pic == 1){
        texture = PIXI.Texture.from(imagePathTrain);
    }else if (pic==2){
        texture = PIXI.Texture.from(imagePathPlane);
    }else{
        texture = PIXI.Texture.from(imagePathTruck);
    }

    // Create a sprite using the texture
    const imageSprite = new PIXI.Sprite(texture);

    imageSprite.width = 25;
    imageSprite.height  = 25;

    // Optionally, set the anchor point to the center (0.5 for both x and y axis)
    imageSprite.anchor.set(0.5);

    // Set the position of the sprite
    imageSprite.x = x;
    imageSprite.y = y;

    return imageSprite;
}

const icons = [];
for (let x=0; x<20; x++){
    const imageSprite = createImageSprite(0, 0); // Create image sprite at position (200, 150)
    icons.push(imageSprite);
    app.stage.addChild(imageSprite); // Add the image sprite to the stage
}

const Dicons = [];
for (let x=0; x<20; x++){
    const imageSprite = createImageSpriteD(0, 0); // Create image sprite at position (200, 150)
    Dicons.push(imageSprite);
    app.stage.addChild(imageSprite); // Add the image sprite to the stage
}


function updateDynamicTasks(data){
    for (const [key, value] of Object.entries(data)) {
        icons[key].y = value["y"]*10;
        icons[key].x = value["x"]*10;
        Dicons[key].y = value["yd"]*10;
        Dicons[key].x = value["xd"]*10;
        
    }
}






// Function to create LIDAR lines
function createLidarLines(car, numLines, maxLength) {
    const lidarLines = new PIXI.Graphics();
    const angleStep = (Math.PI * 2) / numLines;

    for (let i = 0; i < numLines; i++) {
        const angle = i * angleStep;
        const endX = car.x + Math.cos(angle + car.rotation) * maxLength;
        const endY = car.y + Math.sin(angle + car.rotation) * maxLength;
        
        lidarLines.lineStyle(1, 0xFFFF00, 0.5);
        lidarLines.moveTo(car.x, car.y);
        lidarLines.lineTo(endX, endY);
    }

    return lidarLines;
}

// Function to check line intersection
function lineIntersection(x1, y1, x2, y2, x3, y3, x4, y4) {
    const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (den === 0) return null;

    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
    const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;

    if (t > 0 && t < 1 && u > 0) {
        return {
            x: x1 + t * (x2 - x1),
            y: y1 + t * (y2 - y1)
        };
    }

    return null;
}

// Function to update LIDAR lines with intersection checks
function updateLidarLines(car, lidarLines, obstacles) {
    lidarLines.clear();
    const numLines = 5;
    const maxLength = 50;
    const angleStep = (Math.PI * 2) / numLines;

    for (let i = 0; i < numLines; i++) {
        // const angle = car.rotation - 40 + (80 / (numLines - 1)) * i;
        const angle = car.rotation - 40 + 20 * i; 
        let endX = car.x + Math.cos(angle) * maxLength;
        let endY = car.y + Math.sin(angle) * maxLength;

        // Check intersections with obstacles
        for (const obstacle of obstacles) {
            const intersection = lineIntersection(
                car.x, car.y, endX, endY,
                obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2
            );

            if (intersection) {
                endX = intersection.x;
                endY = intersection.y;
            }
        }

        lidarLines.lineStyle(1, 0xFFFF00, 0.5);
        lidarLines.moveTo(car.x, car.y);
        lidarLines.lineTo(endX, endY);
    }
}