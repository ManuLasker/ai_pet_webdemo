import os

DEBUG = os.environ.get('DEBUG', False)
REDIS_HOST = os.environ.get('REDIS_HOST')
TITLE = 'Ai Pet Web app'
DESCRIPTION = 'Web demo app for ai pet semi automatic process'
MODELS_PATH = '/opt/models'
MASK_MODEL_PATH = os.path.join(MODELS_PATH, 'torch.model_segmentation.pt')