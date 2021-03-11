from pydantic import BaseModel

class cacheRequest(BaseModel):
    connection_id: str
    base64_image: str
    name_image: str