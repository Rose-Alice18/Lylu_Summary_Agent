import socket
from fastapi import FastAPI

from code_1 import process_audio
from src.schema import AudioInput,AudioResponse

app = FastAPI(title="Lylu Chat Agent",
              version='0.1.0',
              )

@app.on_event("startup")
async def startup_event():
    """Debug startup event to identify DNS issues"""
    print("🚀 FastAPI startup event triggered...")
    
    # Test basic DNS resolution
    test_domains = [
        "api.openai.com",
        "api.assemblyai.com", 
        "google.com"
    ]
    
    print("🌐 Testing DNS resolution...")
    for domain in test_domains:
        try:
            ip = socket.gethostbyname(domain)
            print(f"✅ {domain} -> {ip}")
        except socket.gaierror as e:
            print(f"❌ {domain} -> DNS Error: {e}")
            

@app.post('/analyse_audio',response_model=AudioResponse)
async def generate_summaray(audio_path:AudioInput):
    result = process_audio(audio_path.audio_path)
    return result