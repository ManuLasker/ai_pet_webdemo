import torch
from redis import Redis
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import logger
from app.config.variables import (DEBUG, REDIS_HOST,
                                  TITLE, DESCRIPTION, MASK_MODEL_PATH)
from app.blending.segmentation import Predictor
from app.blending.models import VGG16_Model, MeanShift
# preconfig models
model = VGG16_Model()
del(model)
# config predictor
Predictor.set_config(model_path=MASK_MODEL_PATH)
Predictor.predict(x = torch.randn(1, 3, 224, 224))

app = FastAPI(debug=DEBUG,
              title=TITLE,
              description=DESCRIPTION)

def get_redis_db():
    db = Redis(host=REDIS_HOST)
    try:
        yield db
    except:
        del db
        
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/templates', StaticFiles(directory='templates'), name='templates')
app_templates = Jinja2Templates(directory='templates')
app_templates.env.globals.update(zip=zip)

from app import routes

