import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from clients.spotify_client import SpotifyClient
from inference.streaming import chunked_inference

app = FastAPI()

# Mount frontend static files under '/static'
app.mount(
    "/static", StaticFiles(directory="frontend", html=True), name="static"
)

# Serve index.html on root
@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(os.getcwd(), 'frontend', 'index.html')
    with open(path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

# Initialize Spotify client (optional)
sp = SpotifyClient()

# Endpoint for Spotify URI ingestion (optional)
@app.post('/ingest/spotify')
async def ingest_spotify(track_uri: str = Form(...)):
    # Extract Spotify track ID
    tid = track_uri.split(':')[-1]
    # Get preview URL and generate lyrics
    preview_url = sp.get_preview_url(tid)
    if preview_url is None:
        return {'error': 'No preview available for this track.'}
    lyrics = chunked_inference(preview_url)
    return { 'generated': lyrics }

# Endpoint for full-track upload
@app.post('/ingest/upload')
async def ingest_upload(file: UploadFile):
    # Ensure raw data directory exists
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    # Check for missing filename
    if not file.filename:
        return {'error': 'No filename provided for uploaded file.'}
    # Save uploaded file
    save_path = os.path.join(raw_dir, file.filename)
    with open(save_path, 'wb') as out_file:
        out_file.write(await file.read())
    # Generate lyrics from the saved audio file
    lyrics = chunked_inference(save_path)
    return { 'generated': lyrics }

# To run the server:
# uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000