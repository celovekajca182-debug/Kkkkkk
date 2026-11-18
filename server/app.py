import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledge import KnowledgeBase
from model import TinyCharRNN

app = FastAPI(title="SWIN AI", version="0.2.0")
model = None
model_path = Path(os.getenv("SWIN_MODEL_PATH", "model.npz"))
knowledge_path = Path(os.getenv("SWIN_KNOWLEDGE_PATH", "data/knowledge.json"))
knowledge = KnowledgeBase(knowledge_path)

if model_path.is_file():
    try:
        model = TinyCharRNN.load(str(model_path))
        print(f"SWIN model loaded: {model_path}")
    except Exception as error:
        print(f"SWIN model load error: {error}")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/")
def root():
    return {
        "name": "SWIN AI",
        "status": "online",
        "model_loaded": model is not None,
        "knowledge_entries": knowledge.size,
        "version": app.version,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "knowledge_entries": knowledge.size,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Сообщение не должно быть пустым.")

    match = knowledge.search(message)
    if match is not None:
        return {
            "answer": match["answer"],
            "mode": "local_knowledge",
            "source": match["title"],
            "confidence": round(match["score"], 3),
        }

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="В локальной базе пока нет ответа на этот вопрос, а веса нейросети ещё не обучены.",
        )

    prompt = message[-160:]
    generated = model.generate(prompt, length=220, temperature=0.75)
    answer = generated[len(prompt):].strip()
    if not answer:
        answer = "Модель пока не смогла продолжить текст. Добавь больше качественных примеров в обучающий корпус."
    return {
        "answer": answer,
        "mode": "character_rnn",
        "source": "собственные обученные веса",
        "confidence": None,
    }
