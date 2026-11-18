import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import TinyCharRNN

app=FastAPI(title="SWIN AI",version="0.1.0")
model=None
path=os.getenv("SWIN_MODEL_PATH","model.npz")
if os.path.exists(path):
    try: model=TinyCharRNN.load(path)
    except Exception as e: print("Model load error:",e)

class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=2000)

@app.get("/")
def root(): return {"name":"SWIN AI","status":"online","model_loaded":model is not None}

@app.get("/health")
def health(): return {"ok":True,"model_loaded":model is not None}

@app.post("/chat")
def chat(req:ChatRequest):
    if model is None: raise HTTPException(503,"Модель ещё не обучена: сначала создай model.npz.")
    answer=model.generate(req.message.strip(),length=220)
    return {"answer":answer[len(req.message.strip()):].strip()}
