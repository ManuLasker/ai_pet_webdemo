var canvas = null;
var ctx = null;
var id = null;

// num click target to watch over rectangle draw if pass > 1 reset square
var numClickTarget = 0;

// to save img.src
var sourceImage = '', targetImage = '';

//for rectangle draw
var xRect0=0, yRect0=0, xRectf=0, yRectf = 0;

function buttonOnChange(buttonObject){
    id = buttonObject.id
    switch (id){
        case 'uploadSourceImage':
            uploadImage(buttonObject, 'sourceImage')
            break;
        case 'uploadTargetImage':
            uploadImage(buttonObject, 'targetImage')
            break;
    }
}

function buttonOnClick(buttonObject){
    id = buttonObject.id
    switch (id){
        case 'predictMask':
            break;
        case 'blendImages':
            break;
    }
}

function setCanvasConfig(canvasObject){
    canvas = document.getElementById(canvasObject.id)
    ctx = canvas.getContext('2d')
}

function uploadImage(buttonObject, canvasId){
    canvas = document.getElementById(canvasId)
    ctx = canvas.getContext('2d')
    if ( buttonObject.files && buttonObject.files[0] ) {
        var FR= new FileReader();
        FR.onload = function(e) {
            var img = new Image();
            img.src = e.target.result;
            console.log('Caching and preprocessing images');
            fetch('/preprocess_image', {
                method: 'POST',
                body: JSON.stringify({
                    base64_image: img.src,
                    name_image: canvasId
                })
            });
            if (canvasId == 'sourceImage'){
                sourceImage = img.src
            }else{
                targetImage = img.src
            }
            img.onload = function() {
                canvas.width = this.width;
                canvas.height = this.height;
                ctx.drawImage(this, 0, 0,
                    this.width, this.height);
            };
        };       
        FR.readAsDataURL(buttonObject.files[0]);
    }
}

function mouseDown(event, canvasObject){
    console.log(canvasObject)
    setCanvasConfig(canvasObject)
    draw('down', event, canvasObject, canvas, ctx)
}

function mouseMove(event, canvasObject){
    setCanvasConfig(canvasObject)
    draw('move', event, canvasObject, canvas, ctx)
}

function mouseUp(event, canvasObject){
    setCanvasConfig(canvasObject)
    draw('up', event, canvasObject, canvas, ctx)
}

function drawRect(canvas, ctx){
    var rectWidth = Math.abs(xRect0 - xRectf)
    var rectHeight = Math.abs(yRect0 - yRectf)
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 5;
    ctx.strokeRect(xRect0, yRect0, rectWidth, rectHeight);
    ctx.stroke()

    fetch('/save_coordinates', {
        method: 'POST',
        body: JSON.stringify({
            box_xyxy_coords: [xRect0, yRect0, xRectf, yRectf]
        })
    });
}

function draw(action, event, canvasObject, canvas, ctx){
    id = canvasObject.id
    switch(action){
        case 'down':
            if (canvasObject.id == 'sourceImage'){

            }else if (canvasObject.id == 'targetImage'){
                //square
                xRect0 = event.clientX - canvas.getBoundingClientRect().left;
                yRect0 = event.clientY -  canvas.getBoundingClientRect().top;
                
                if (numClickTarget > 1){
                    
                    if (targetImage != ''){
                        var img = new Image();
                        img.src = targetImage;
                        img.onload = function() {
                            canvas.width = this.width;
                            canvas.height = this.height;
                            ctx.drawImage(this, 0, 0,
                                this.width, this.height);
                        };
                    }else{
                        ctx.fillStyle = 'aqua'
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                    }
                    numClickTarget = 0;
                }

                numClickTarget ++;
                console.log(xRect0, yRect0)
            }
            break;
        case 'up':
            if (canvasObject.id == 'sourceImage'){

            }else if (canvasObject.id == 'targetImage'){
                xRectf = event.clientX - canvas.getBoundingClientRect().left
                yRectf = event.clientY - canvas.getBoundingClientRect().top
                numClickTarget ++;
                console.log(xRectf, yRectf)
                drawRect(canvas, ctx)
            }
            break;
        case 'move':
            if (canvasObject.id == 'sourceImage'){

            }else if (canvasObject.id == 'targetImage'){
                
            }
            break;
    }
}