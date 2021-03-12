from pydantic import BaseModel
from typing import List

class cacheImageRequest(BaseModel):
    base64_image: str
    name_image: str
    
class cacheCoordinatesPositioningRequest(BaseModel):
    box_xyxy_coords: List[float]