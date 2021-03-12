import json
from fastapi import templating
from fastapi import Request, Depends
from app.routes_models import (cacheImageRequest,
                               cacheCoordinatesPositioningRequest)
from app import app, app_templates, get_redis_db, Redis, logger
from fastapi.responses import HTMLResponse
from PIL import Image

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return app_templates.TemplateResponse('index.html', {'request': request})

@app.post('/preprocess_image', description='Save Image base64 decode from frontend')
def preprocess_image(body: cacheImageRequest, db: Redis = Depends(get_redis_db)):
    logger.info('Saving images')
    db.set(body.name_image, body.base64_image)
    return {'status': 'ok'}

@app.post('/save_coordinates', description='Save coordinates blending and positioning to redis db')
def save_coordinates(body: cacheCoordinatesPositioningRequest,
                     db: Redis = Depends(get_redis_db)):
    logger.info('Saving xyxy coordinates')
    db.set('xyxy_coordinates',  json.dumps(body.box_xyxy_coords))
    return 'ok'