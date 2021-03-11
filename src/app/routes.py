from fastapi import templating
from fastapi import Request, Depends
from app.routes_models import cacheRequest
from app import app, app_templates, get_redis_db, Redis
from fastapi.responses import HTMLResponse

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return app_templates.TemplateResponse('index.html', {'request': request})

@app.post('/cache_image', tags=['Redis db'])
def cache_image(body: cacheRequest, db: Redis = Depends(get_redis_db)):
    print(body.dict())
    return 'ok'