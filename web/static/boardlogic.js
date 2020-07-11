var namespace = '/tablut';
var gameBoard = document.getElementById('gameBoard');

var socket = io(namespace);
ScrollReveal().reveal('#gameBoard');

var state = null;
var tableroObjects = {}
var moveCoordinates; // Move ID (int) -> Move From Pos to Pos [x1, y1, x2, y2]
var coordinatesMove; // Move From Pos to Pos [x1, y1, x2, y2] -> Move ID (int)
var movePerPiece;   // Coordinate [x1, y1] -> Array[ [x2, y2] ]
var perspectiva = false;

socket.on('connect', function() {
    socket.emit('my event', {data: 'I\'m connected!'});
});

function alternar() {
    perspectiva = !perspectiva;
    if (perspectiva) {
        // gameBoard.style.webkitTransform=`perspective(800px) rotateX(${38}deg) rotateY(${0}deg) rotateZ(${-45}deg)`;
        gameBoard.style.webkitTransform=`rotateX(45deg) rotateY(${0}deg) rotateZ(${-45}deg)`;
        gameBoard.style.boxShadow = "-40px 40px 10px 2px rgba(102, 75, 75, 0.1)";
        var x = document.getElementsByClassName('ficha');
        console.info(x);
        for(let i = 0; i < x.length; i++) {
            // x[i].style.webkitTransform=`perspective(800px) rotateX(${45}deg) rotateY(${0}deg) rotateZ(${0}deg)`;
            x[i].style.webkitTransform="translateZ(10px) rotateX(-45deg) rotateZ(36deg)";
        }
        //x.forEach(img => {
        //    img.style.webkitTransform=`perspective(800px) rotateX(${20}deg) rotateY(${0}deg) rotateZ(${0}deg)`;
        //});

    } else {
        gameBoard.style.webkitTransform=`translateY(80px)perspective(800px) rotateX(${0}deg) rotateY(${0}deg) rotateZ(${0}deg)`;
        gameBoard.style.boxShadow = "";
        var x = document.getElementsByClassName('ficha');
        for(let i = 0; i < x.length; i++) {
            x[i].style.webkitTransform=`perspective(800px) rotateX(${0}deg) rotateY(${0}deg) rotateZ(${0}deg)`;
        }
    }
}

function on_drag_started(object) {
    object.srcElement.style.opacity = 0.3;
    console.info('dragstart', object);
    on_click_cell(object);
}

function on_dragover(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
    console.info(e);
}

function on_drop(object) {
    if (object.stopPropagation) 
        object.stopPropagation();
    
    console.info(object, "ondrop");
}

function on_drag_ended(ev) {
    console.info(ev);
    on_click_cell(ev);
    if (ev.srcElement == ev.toElement) {
        ev.srcElement.style.opacity = 1;
        console.info('son el mismo target');
    }

}

function on_click_cell(e) {
    movement_completed = false;
    if (e.srcElement.parentElement.classList.contains('destination')) {
        console.info('soy dest de un source...');
        var sel = document.getElementsByClassName("selected");
        if (sel[0] !== null && sel[0] !== undefined) {
            let src = sel[0];
            let [xsrc, ysrc] = src.getAttribute("cell").split("_");
            let [xdest, ydest] = e.srcElement.getAttribute("cell").split("_");
            let mvmnt = coordinatesMove[[xsrc, ysrc, xdest, ydest].map((e) => Number.parseInt(e))];
            if (mvmnt !== undefined) {
                movement_completed = true;
                socket.emit('move', {'movement': mvmnt, 'state': state})
                console.info('es legal, comprobacion de frontend que no sustituye a otra en el backend, pero de momento yo lo alloweo pls dont hack');
                // cirujia
                let traspasamos = src.innerHTML;
                let aqui = e.srcElement.innerHTML;
                src.innerHTML = aqui;
                aqui.innerHTML = traspasamos;
                

            }
        }
        
        

    } 

    Object.values(tableroObjects).forEach((el) => {
        if (el.classList.contains('selected'))
            el.classList.remove('selected');
        
        if (el.parentElement.classList.contains('destination')) {
            el.parentElement.classList.remove('destination');
            tableroObjects[[x2, y2]].setAttribute("draggable", 'false');
        }

        
        
    })

   if (!movement_completed) {
    e.srcElement.classList.toggle('selected');
    let [x1, x2] = e.srcElement.getAttribute('cell').split("_");
    if (movePerPiece[[x1, x2]] !== undefined) {
        movePerPiece[[x1, x2]].forEach(([x2, y2]) => {
            tableroObjects[[x2, y2]].parentElement.classList.add('destination');
            tableroObjects[[x2, y2]].setAttribute("draggable", 'true');
            // tableroObjects[[x2, y2]].parentElement.style.backgroundColor = 'red';
        });
    }
   }

    
}

function create_board() {
    for (let i = 0; i < 9; i++) {
        var tr = document.createElement('tr');
        for (let j = 0; j < 9; j++) {
            let z = document.createElement('img');

            let p = document.createElement('td');
            let d = document.createElement("div");
            d.setAttribute("class", "ficha");
            d.setAttribute("draggable", "true");
            d.addEventListener('click', on_click_cell, false);
            d.addEventListener('dragstart', on_drag_started, false);
            d.addEventListener('dragend', on_drag_ended, false);
            d.addEventListener('dragover', on_dragover, false);
            d.addEventListener('dragdrop', on_drop, false);
            d.setAttribute("cell", `${i}_${j}`);
            tableroObjects[[i, j]] = d;
            z.setAttribute("xd", `${i}_${j}`);
            d.appendChild(z);
            p.appendChild(d);
            tr.appendChild(p);
        }
        gameBoard.appendChild(tr);    
    }
    // gameBoard.style.webkitTransform=`translateX(-50%) rotateX(45deg) rotateY(${0}deg) rotateZ(${-45}deg)`;
    // gameBoard.style.transformOrigin = "50% 0";
    // gameBoard.style.opacity = 0;
    // gameBoard.style.boxShadow = "-20px 20px 10px 2px rgba(102, 75, 75, 0.1)";
    
}
create_board();

socket.on('start_game', function(msg, cb) {
    apply_move(msg.data);
            // gameBoard.innerHTML += `<td>${el}</td>`);
    //
    //socket.emit('move', {data: {'state': state.outstr, 'stats': state.stats, 'move': moves[0]}});
    console.log(cb);
});

function apply_move(stateData) {
    let tabla = document.getElementById('gameBoard');
    state = JSON.parse(stateData);
    

    for (let i = 0; i < tabla.children.length; i++) {
        let row = tabla.children[i];
        for (let j = 0; j < row.children.length; j++) {
            let val = state.outstr[i][j];
            ; // TODO VOY POR AQUI, Uncaught TypeError: Cannot set property 'innerText' of undefined
            // APARTIR DE LA SEGUNDA LINEA
            if (val == 0) {
                row.children[j].children[0].innerText = '';
            } else {
                row.children[j].children[0].innerText = val;
            }
            
        }
    }
    
    parseMovement(state.moves)
    
    //gameBoard.style.animation=`translateY(100px)`;
    // gameBoard.style.webkitTransform=`perspective(800px) rotateX(${42}deg) rotateY(${0}deg) rotateZ(${-45}deg)`;
    // gameBoard.style.webkitTransform=`translateX(-50%) rotateX(45deg) rotateY(${0}deg) rotateZ(${-45}deg)`;
    // gameBoard.style.boxShadow = "-40px 40px 20px 5px rgba(0, 0, 80, 0.2)";
}

function parseMovement(movesObject) {
    moveCoordinates  = movesObject;
    coordinatesMove  = {};
    movePerPiece = {};
    console.info(moveCoordinates);
    Object.entries(moveCoordinates).forEach(([idx, e]) => {
        let [a, v] = e;

        coordinatesMove[v] = a;
        [x1, y1, x2, y2] = v;
        if (movePerPiece.hasOwnProperty([x1, y1])) {
            movePerPiece[[x1,y1]].push([x2, y2]);
        } else {
            movePerPiece[[x1, y1]] = [[x2, y2]];
        }
    });
}

socket.on('move', (msg, cb) => {
    apply_move(msg.data)
})