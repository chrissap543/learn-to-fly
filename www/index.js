import * as sim from "lib-simulation-wasm";

const simulation = new sim.Simulation();
const world = simulation.world(); 

document.getElementById('train').onclick = function() {
    console.log(simulation.train()); 
}; 

const viewport = document.getElementById('viewport');
const viewportWidth = viewport.width;
const viewportHeight = viewport.height; 
const viewportScale = window.devicePixelRatio || 1; 

viewport.width = viewportWidth * viewportScale;
viewport.height = viewportHeight * viewportScale;

viewport.style.width = viewportWidth + 'px'; 
viewport.style.height = viewportHeight + 'px'; 

const ctxt = viewport.getContext('2d'); 
ctxt.scale(viewportScale, viewportScale); 
ctxt.fillStyle = 'rgb(0, 0, 0)';

CanvasRenderingContext2D.prototype.drawTriangle =
    function (x, y, size, rotation) {
        this.beginPath();

        this.moveTo(
            x - Math.sin(rotation) * size * 1.5,
            y + Math.cos(rotation) * size * 1.5,
        );

        this.lineTo(
            x - Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
            y + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x - Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
            y + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x - Math.sin(rotation) * size * 1.5,
            y + Math.cos(rotation) * size * 1.5,
        );

        this.stroke();

        this.fillStyle = 'rgb(255, 255, 255)'; 
        this.fill(); 
    };
CanvasRenderingContext2D.prototype.drawCircle = 
    function(x, y, radius) {
        this.beginPath(); 

        this.arc(x, y, radius, 0, 2.0 * Math.PI); 

        this.fillStyle = 'rgb(0, 255, 128)'; 
        this.fill(); 
    }


let lastTime = 0; 
const targetFPS = 30; 
const frameInterval = 1000 / targetFPS; 
function redraw(currentTime) {
    if (currentTime - lastTime >= frameInterval) {
        ctxt.clearRect(0, 0, viewportWidth, viewportHeight); 
        simulation.step(); 

        const world = simulation.world(); 

        for (const food of world.foods) {
            ctxt.drawCircle(
                food.x * viewportWidth,
                food.y * viewportHeight,
                (0.01 / 2.0) * viewportWidth, 
            ); 
        }

        for (const animal of world.animals) { 
            ctxt.drawTriangle(
                animal.x * viewportWidth, 
                animal.y * viewportHeight, 
                0.01 * viewportWidth, 
                animal.rotation
            ); 
        }
        lastTime = currentTime; 
    }
    requestAnimationFrame(redraw); 
}

redraw(0); 
