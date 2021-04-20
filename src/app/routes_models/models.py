from pydantic import BaseModel
from typing import List

class cacheImageRequest(BaseModel):
    base64_image: str
    name_image: str
    
class cacheCoordinatesPositioningRequest(BaseModel):
    box_xyxy_coords: List[float]
    name_coords: str

class grabCutRequest(BaseModel):
    mask_helper: str
    
class maskModelRequest(BaseModel):
    name_image: str
    
class blendModelRequest(BaseModel):
    naive: bool