import uvicorn
from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import glob
import subprocess
import signal
import pyaudio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients: List[WebSocket] = []
matcher_process = None

class StartRequest(BaseModel):
    song_file: str
    params: dict
    mic_device: Optional[int] = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New websocket connected!")
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            # We don't expect messages from client, but keep conn alive
            await websocket.receive_text()
    except Exception as e:
        print("Websocket disconnected:", e)
        if websocket in clients:
            clients.remove(websocket)

@app.post("/transition")
async def transition(request: Request):
    data = await request.json()
    print(f"Broadcasting transition: {data}")
    dead_clients = []
    for client in clients:
        try:
            await client.send_json(data)
        except Exception:
            dead_clients.append(client)
    
    for c in dead_clients:
        if c in clients:
            clients.remove(c)
        
    return {"status": "ok"}

@app.get("/songs")
async def get_songs():
    songs = []
    txt_files = glob.glob("*.txt")
    
    # Exclude non-song txt files
    exclude = ["requirements.txt", "lyrics.txt", "test_out.txt", "test_output_smoke.txt", "test_output_v2.txt", "test_output_v2b.txt"]
    for f in txt_files:
        if f in exclude or "answers" in f or "transcript" in f:
            continue
            
        try:
            with open(f, "r") as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            title = f.replace(".txt", "").replace("_", " ").title()
            songs.append({
                "id": f,
                "title": title,
                "filename": f,
                "lyrics": lines
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return {"songs": sorted(songs, key=lambda x: x["title"])}

@app.get("/mics")
async def get_mics():
    p = pyaudio.PyAudio()
    mics = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            mics.append({
                "id": i,
                "name": dev.get('name')
            })
    p.terminate()
    return {"mics": mics}

@app.post("/start")
async def start_matcher(req: StartRequest):
    global matcher_process
    
    if matcher_process and matcher_process.poll() is None:
        matcher_process.send_signal(signal.SIGTERM)
        matcher_process.wait()
        
    cmd = [
        "python", "matcher.py", req.song_file,
        "--trigger-words", str(req.params.get("trigger_words", 5)),
        "--min-match-pct", str(req.params.get("min_match_pct", 60)),
        "--min-time-on-line", str(req.params.get("min_time_on_line", 4.0)),
        "--word-tolerance", str(req.params.get("word_tolerance", 2)),
        "--update-interval", str(req.params.get("update_interval", 0.5))
    ]
    
    if req.mic_device is not None:
        cmd.extend(["--mic-device", str(req.mic_device)])
        
    print(f"Starting matcher: {' '.join(cmd)}")
    matcher_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    return {"status": "started"}

@app.post("/advance")
async def manual_advance():
    global matcher_process
    if matcher_process and matcher_process.poll() is None:
        try:
            matcher_process.stdin.write(b"NEXT\n")
            matcher_process.stdin.flush()
            return {"status": "advanced"}
        except Exception as e:
            return {"status": f"error: {e}"}
    return {"status": "not running"}

@app.post("/stop")
async def stop_matcher():
    global matcher_process
    if matcher_process and matcher_process.poll() is None:
        matcher_process.send_signal(signal.SIGTERM)
        matcher_process.wait()
        matcher_process = None
        return {"status": "stopped"}
    return {"status": "not running"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9000, reload=True)
