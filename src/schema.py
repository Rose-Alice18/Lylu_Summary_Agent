from typing import Any, Dict
from pydantic import BaseModel

class AudioInput(BaseModel):
    audio_path:str
    
class AudioResponse(BaseModel):
    success: bool
    dev_message:str
    user_message:str
    payload: Dict[str,Any]


