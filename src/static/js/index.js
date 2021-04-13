var canvas = null;
var ctx = null;
var id = null;
var color = null;

//temporal mask canvas
var tempCanvas = document.createElement('canvas', {id:'tempCanvas'})
var tempCtx = tempCanvas.getContext('2d')
tempCanvas.width = 224 //default
tempCanvas.height = 224 //default
var tempColor = null;

//get drop downs
var selectObject = document.getElementById('colors');
var strokeObject = document.getElementById('strokes');
var methodObject = document.getElementById('methods')

// num click target to watch over rectangle draw if pass > 1 reset square
var numClickTarget = 0;
var isChange = false;

// to save img.src
var sourceImage = '', targetImage = '';

//for rectangle draw
var xRect0=0, yRect0=0, xRectf=0, yRectf = 0;
//for path drawing
var xPrev=0, yPrev=0, xCurr=0, yCurr=0;
var flag = false, dot_flat = false;

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
        case 'previewImages':
            break;
    }
}

function setCanvasConfig(canvasObject){
    canvas = document.getElementById(canvasObject.id)
    ctx = canvas.getContext('2d')
}

function uploadImage(buttonObject, canvasId){
    var canvas_ = document.getElementById(canvasId)
    var ctx_ = canvas_.getContext('2d')
    if ( buttonObject.files && buttonObject.files[0] ) {
        var FR = new FileReader();
        FR.onload = function(e) {
            var img = new Image();
            // console.log('Caching and preprocessing images');
            fetch('/cache_original_images', {
                method: 'POST',
                body: JSON.stringify({
                    base64_image: e.target.result,
                    name_image: canvasId
                })
            }).then(response => response.json())
            .then(response => {
                // console.log(response)
                img.src = response.imageBase64;
                // img.src =  e.target.result;

                if (canvasId == 'sourceImage'){
                    sourceImage = img.src
                }else{
                    targetImage = img.src
                }
                img.onload = function() {
                    canvas_.width = this.width;
                    canvas_.height = this.height;
                    if (canvasId == 'sourceImage'){
                        tempCanvas.width = this.width;
                        tempCanvas.height = this.height;
                        tempCtx.fillStyle = 'gray'
                        //tempCtx.fillStyle = `rgba(255,255,255,0)`
                        tempCtx.fillRect(0, 0, this.width, this.height);
                        // console.log(tempCanvas)
                    }
                    ctx_.drawImage(this, 0, 0,
                        this.width, this.height);
                };
            });
        };       
        FR.readAsDataURL(buttonObject.files[0]);
    }
}

function mouseDown(event, canvasObject){
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

function mouseOut(event, canvasObject){
    setCanvasConfig(canvasObject);
    draw('out', event, canvasObject, canvas, ctx)
}

function drawRect(canvas, ctx, name, stroke){
    var rectWidth = Math.abs(xRect0 - xRectf)
    var rectHeight = Math.abs(yRect0 - yRectf)
    ctx.strokeStyle = 'green';
    ctx.lineWidth = stroke;
    ctx.strokeRect(xRect0, yRect0, rectWidth, rectHeight);
    // ctx.stroke()

    // console.log(xRect0, yRect0, xRectf, yRectf)
    // console.log('caching target coordinates')
    fetch('/cache_coordinates', {
        method: 'POST',
        body: JSON.stringify({
            box_xyxy_coords: [xRect0, yRect0, xRectf, yRectf],
            name_coords: name
        })
    });
}

function setPath(event, canvas){
    xPrev = xCurr;
    yPrev = yCurr;
    // Current x, y 
    xCurr = event.clientX - canvas.getBoundingClientRect().left;
    yCurr = event.clientY - canvas.getBoundingClientRect().top;
}

function beginRectDraw(event, canvas, srcImage, ctx){
    xRect0 = event.clientX - canvas.getBoundingClientRect().left;
    yRect0 = event.clientY -  canvas.getBoundingClientRect().top;
    
    if (numClickTarget > 1){
        if (srcImage != ''){
            var img = new Image();
            img.src = srcImage;
            img.onload = function() {
                canvas.width = this.width;
                canvas.height = this.height;
                ctx.drawImage(this, 0, 0,
                    this.width, this.height);
            };
        }else{
            ctx.fillStyle = 'black'
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        numClickTarget = 0;
    }

    numClickTarget ++;
    // console.log(xRect0, yRect0)
}

function finishRectDraw(event, canvas, ctx, name, stroke){
    xRectf = event.clientX - canvas.getBoundingClientRect().left
    yRectf = event.clientY - canvas.getBoundingClientRect().top
    numClickTarget ++;
    // console.log(xRectf, yRectf)
    drawRect(canvas, ctx, name, stroke)
}

function drawDotInTempCanvas(color, xCurr, yCurr, stroke){
    tempCtx.beginPath();
    tempCtx.fillStyle = color;
    tempCtx.arc(xCurr, yCurr, stroke*0.5, 0, 2*Math.PI)
    tempCtx.fill()
}

function drawMoveInTempCanvas(stroke, color, xPrev, yPrev, xCurr, yCurr){
    tempCtx.beginPath();
    tempCtx.moveTo(xPrev, yPrev)
    tempCtx.lineTo(xCurr, yCurr)
    tempCtx.lineCap = "round";
    tempCtx.lineJoin = "round";
    tempCtx.strokeStyle = color;
    tempCtx.lineWidth = stroke;
    tempCtx.stroke()
    tempCtx.closePath()
}


function draw(action, event, canvasObject, canvas, ctx){
    id = canvasObject.id
    var stroke = strokeObject.value;
    var color = selectObject.value;
    var method = methodObject.value;
    switch(action){
        case 'down':
            if (canvasObject.id == 'sourceImage'){
                if (method == 'mask'){
                    setPath(event, canvas)
                    flag = true, dot_flag = true;
                    if (dot_flag){
                        ctx.beginPath();
                        ctx.fillStyle = color;
                        ctx.arc(xCurr, yCurr, stroke*0.5, 0, 2*Math.PI)
                        ctx.fill()
                        // ctx.closePath()
                        drawDotInTempCanvas(color, xCurr, yCurr, stroke)
                        dot_flag = false;
                    }
                }else{
                    beginRectDraw(event, canvas, sourceImage, ctx)
                }
                isChange=true
            }else if (canvasObject.id == 'targetImage'){
                beginRectDraw(event, canvas, targetImage, ctx)
            }
            break;
        case 'up':
            if (canvasObject.id == 'sourceImage'){
                flag = false;
                if (method != 'mask'){
                    finishRectDraw(event, canvas, ctx, 'sourceImage', stroke)
                    var img = new Image()
                    var imgPreview = new Image()
                    fetch('/mask_grabcut_square_helper',
                    {
                        method: 'POST'
                    }).then(response => response.json())
                    .then(response => {
                        // console.log(response)
                        var canvasCut = document.getElementById('grabCutImage')
                        var ctxCut = canvasCut.getContext('2d')
                        img.src = response.mask_rect
                        img.onload = function() {
                            canvasCut.width = this.width
                            canvasCut.height = this.height
                            ctxCut.drawImage(this, 0, 0,
                                this.width, this.height)
                        }
                        // upload image to preview image canvas
                        var canvasPreview = document.getElementById('previewImage')
                        var ctxPreview = canvasPreview.getContext('2d')
                        imgPreview.src = response.preview_cut
                        imgPreview.onload = function() {
                            canvasPreview.width = this.width
                            canvasPreview.height = this.height
                            ctxPreview.drawImage(this, 0, 0, this.width, this.height)
                        }
                    })
                }else{
                    var manualMaskHelperURL = tempCanvas.toDataURL()
                    console.log(manualMaskHelperURL)
                    var img = new Image()
                    var imgPreview = new Image()
                    fetch('/mask_wm_grabcut_helper', 
                    {
                        method: 'POST',
                        body: JSON.stringify({
                            mask_helper: manualMaskHelperURL
                        })
                    }).then(response => response.json())
                    .then(response => {
                        console.log(response)
                        var canvasCut = document.getElementById('grabCutImage')
                        var ctxCut = canvasCut.getContext('2d')
                        img.src = response.mask
                        img.onload = function() {
                            canvasCut.width = this.width
                            canvasCut.height = this.height
                            ctxCut.drawImage(this, 0, 0,
                                this.width, this.height)
                        }
                        // upload image to preview image canvas
                        var canvasPreview = document.getElementById('previewImage')
                        var ctxPreview = canvasPreview.getContext('2d')
                        imgPreview.src = response.preview_cut
                        imgPreview.onload = function() {
                            canvasPreview.width = this.width
                            canvasPreview.height = this.height
                            ctxPreview.drawImage(this, 0, 0, this.width, this.height)
                        }
                    })
                }

                if (isChange){
                    //call grab cut algorithm
                    isChange = false
                }
            }else if (canvasObject.id == 'targetImage'){
                finishRectDraw(event, canvas, ctx, 'targetImage')
            }
            break;
        case 'move':
            if (canvasObject.id == 'sourceImage'){
                if (flag){
                    setPath(event, canvas)
                    ctx.beginPath();
                    ctx.moveTo(xPrev, yPrev)
                    ctx.lineTo(xCurr, yCurr)
                    ctx.lineCap = "round";
                    ctx.lineJoin = "round";
                    ctx.strokeStyle = color;
                    ctx.lineWidth = stroke;
                    ctx.stroke()
                    ctx.closePath()
                    drawMoveInTempCanvas(stroke, color, xPrev, yPrev, xCurr, yCurr)
                }
            }
            break;
        case 'out':
            flag = false;
            break;
    }
}