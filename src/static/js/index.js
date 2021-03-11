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

function uploadImage(buttonObject, canvasId){
    var canvas = document.getElementById(canvasId)
    var ctx = canvas.getContext('2d')
    if ( buttonObject.files && buttonObject.files[0] ) {
        var FR= new FileReader();
        FR.onload = function(e) {
            var img = new Image();
            img.src = e.target.result;
            console.log('Caching image')
            fetch('/cache_image', {
                method: 'POST',
                body: JSON.stringify({
                    connection_id: 'one',
                    base64_image: img.src,
                    name_image: canvasId
                })
            })
            img.onload = function() {
                canvas.width = 320;
                canvas.height = 320;
                ctx.drawImage(this, 0, 0, 320, 320);
            };
        };       
        FR.readAsDataURL(buttonObject.files[0]);
    }
}
