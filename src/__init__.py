from fastapi import FastAPI

from code_1 import process_audio
from src.schema import AudioInput,AudioResponse

app = FastAPI(title="Lylu Chat Agent",
              version='0.1.0',
              )

@app.post('/analyse_audio',response_model=AudioResponse)
async def generate_summaray(audio_path:AudioInput):
    result = process_audio(audio_path)
    return result