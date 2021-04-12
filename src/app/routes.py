import json
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
    # Load mask done using grab cut square helper if there is one
    mask_square_helper = json.loads(db.get('squareMaskOriginal'))
    # Get sizes
    original_size = json.loads(db.get('sourceImage''_original_size'))
    resized_size = json.loads(db.get('sourceImage''_new_size'))
    # apply grab cut with mask helper algorithm
    
    return {'status': 'ok'}
        
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
    db.set('squareMaskOriginal', mask_b64)
    db.set('squareMaskResized', mask_b64_resized)
    return {'mask_rect': 'data:image/jpeg;base64'+','+mask_b64_resized,
            'preview_cut': 'data:image/jpeg;base64'+','+preview_b64_resized}