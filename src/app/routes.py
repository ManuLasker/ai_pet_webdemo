import json
from app.routes_models.models import blendModelRequest
from app.utils import image as I
from fastapi import Request, Depends
from app.routes_models import *
from app import app, app_templates, get_redis_db, Redis, logger
from fastapi.responses import HTMLResponse

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return app_templates.TemplateResponse('index.html', {'request': request})

@app.post('/cache_coordinates', description='Save coordinates blending and positioning to redis db')
def save_coordinates(body: cacheCoordinatesPositioningRequest,
                     db: Redis = Depends(get_redis_db)):
    logger.info('Saving xyxy coordinates')
    db.set('xyxy_'+body.name_coords+'_coordinates',  json.dumps(body.box_xyxy_coords))
    return {'status': 'ok'}

@app.post('/cache_original_images', description='Preprocess image source and target'
          ' resizing source image to 224 x 224 and target image to 640 x 640'
          ' and caching originals and preprocessing')
def preprocess_image_route(body: cacheImageRequest,
                           db:Redis = Depends(get_redis_db)):
    logger.info('Caching base64 640x640 image encode and this size as the original size')
    # Reduce image to 640x640
    logger.info('image id: ' + body.name_image)
    pil_image = I._base64_to_pil(body.base64_image)
    pil_image = I.resize_with_pad(pil_image, (640, 640))
    base64_image = I._pil_to_base64(pil_image)
    # Cache base64 image string encode
    db.set(body.name_image, json.dumps(base64_image))
    # Base 64 format 
    base64_format = body.base64_image.split(',')[0]
    # Cache original size image
    db.set(body.name_image + '_original_size',
           json.dumps(pil_image.size[::-1]))
    # Resizing original image
    new_shape = (224, 224)
    logger.info('Resizing original image to new image')
    pil_image_resize = I.resize_with_pad(pil_image, new_shape)
    # Cache new size
    db.set(body.name_image + '_new_size', json.dumps(new_shape))
    # Transform pil image to base64 image
    base64_image_resize = I._pil_to_base64(pil_image_resize)
    # Cache resized image
    logger.info('Cache resized image')
    db.set(body.name_image+'Resized', json.dumps(base64_image_resize))
    return {'imageName':  body.name_image,
            'imageBase64': base64_format+','+base64_image_resize}
    
@app.post('/mask_wm_grabcut_helper', description='Apply grabcut algorithm')
def mask_grabcut_wm_helper_route(body: grabCutRequest, db:Redis = Depends(get_redis_db)):
    # Extract mask helper from body request
    base64_format = body.mask_helper.split(',')[0]
    # Convert base64 image encode to pil image
    mask_helper = body.mask_helper 
    # Load source image in base64
    src_image = json.loads(db.get('sourceImage'))
    # Get sizes
    original_size = json.loads(db.get('sourceImage''_original_size'))
    resized_size = json.loads(db.get('sourceImage''_new_size'))
    # apply grab cut with mask helper algorithm
    mask_b64, mask_b64_resized, preview_b64_resized = I.get_grabcut_by_mask(src_image,
                                                                            mask_helper,
                                                                            original_size,
                                                                            resized_size)
    # Cache both paint mask
    db.set('paintMaskOriginal', json.dumps(mask_b64))
    db.set('paintMaskResized', json.dumps(mask_b64_resized))
    db.set('previewPaintMaskCut', json.dumps(preview_b64_resized))
    return {'mask': 'data:image/jpeg;base64'+','+mask_b64_resized,
            'preview_cut': 'data:image/jpeg;base64'+','+preview_b64_resized}
        
@app.post('/mask_grabcut_square_helper', description='Apply grabcut algorithm')
def mask_grabcut_square_helper_route(db:Redis = Depends(get_redis_db)):
    # Get image and xyxy dimensions
    rect_dimensions = json.loads(db.get('xyxy_sourceImage_coordinates'))
    src_image = json.loads(db.get('sourceImage'))
    # Get sizes
    original_size = json.loads(db.get('sourceImage''_original_size'))
    resized_size = json.loads(db.get('sourceImage''_new_size'))
    # Apply grabcut
    logger.info("Applying grab cut algorithm")
    # Rescaling the dimensions 
    mask_b64, mask_b64_resized, preview_b64_resized = I.get_grabcut_by_rect(src_image,
                                                                        rect_dimensions,
                                                                        original_size,
                                                                        resized_size)
    # Cache both square mask
    db.set('squareMaskOriginal', json.dumps(mask_b64))
    db.set('squareMaskResized', json.dumps(mask_b64_resized))
    return {'mask_rect': 'data:image/jpeg;base64'+','+mask_b64_resized,
            'preview_cut': 'data:image/jpeg;base64'+','+preview_b64_resized}
    
@app.post('/mask_model_prediction', description='Predict mask using dl model')
def predict_mask_model(db: Redis = Depends(get_redis_db)):
    # Get source image
    src_image = json.loads(db.get('sourceImage'))
    # Get sizes
    original_size = json.loads(db.get('sourceImage''_original_size'))
    resized_size = json.loads(db.get('sourceImage''_new_size'))
    # predict mask
    mask_b64, mask_b64_resized, preview_b64_resized = I.predict_mask_using_model(src_image,
                                                                                original_size,
                                                                                resized_size)
    # cache all model mask
    db.set('modelMaskOriginal', json.dumps(mask_b64))
    db.set('modelMaskResized', json.dumps(mask_b64_resized))
    db.set('modelPreviewCut', json.dumps(preview_b64_resized))
   
    return {'model_mask': 'data:image/jpeg;base64'+','+mask_b64_resized,
            'preview_cut': 'data:image/jpeg;base64'+','+preview_b64_resized}

@app.post('/blend_image', description='blend images using all the information')
def blend(body: blendModelRequest, db: Redis = Depends(get_redis_db)):
    # load src image information
    src_image = json.loads(db.get('sourceImage'))
    # load mask image information from grabcut
    mask_image = json.loads(db.get('paintMaskOriginal'))
    # load target information for the resized version
    # load target image information
    target_image = json.loads(db.get('targetImage''Resized'))
    # load dims from target blending
    dims = json.loads(db.get('xyxy_'+'targetImage'+'_coordinates'))
    # load dimensions
    # TODO
    # blend images
    b64original_blend, b64resized_blend = I.get_blending_image(src=src_image,
                                                               mask=mask_image,
                                                               target=target_image,
                                                               dims=dims,
                                                               naive=body.naive)
    # save in cache
    db.set('b64original_blend', json.dumps(b64original_blend))
    if body.naive:
        db.set('b64resized_blend_', json.dumps(b64resized_blend))
    else:
        db.set('b64resized_blend', json.dumps(b64resized_blend))
    return {'finish_blend': True,
            'blend_preview': 'data:image/jpeg;base64'+','+b64resized_blend} 
    

@app.get('/get_blend_image', description='get blend image to update view')
def get_blend_image(db: Redis = Depends(get_redis_db)):
    resized_blend_image = db.get('b64resized_blend')
    resized_blend_image = (resized_blend_image if resized_blend_image is None 
                            else 'data:image/jpeg;base64'+','+json.loads(resized_blend_image))
    return {'blend_preview': resized_blend_image}

@app.get('/get_preview_grabcut', description='get preview image for grabcut')
def get_preview_image(db: Redis = Depends(get_redis_db)):
    grabcutb64 = json.loads(db.get('previewPaintMaskCut'))
    return {'preview_image': 'data:image/jpeg;base64'+','+grabcutb64}