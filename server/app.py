import os
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from knowledge import KnowledgeBase
from model import TinyCharRNN

app = FastAPI(title="SWIN AI", version="0.3.0")

rnn_model = None
transformer_model = None
rnn_path = Path(os.getenv("SWIN_MODEL_PATH", "model.npz"))
transformer_path = Path(
    os.getenv("SWIN_TRANSFORMER_PATH", "models/swin-transformer.pt")
)
knowledge_path = Path(os.getenv("SWIN_KNOWLEDGE_PATH", "data/knowledge.json"))
knowledge = KnowledgeBase(knowledge_path)

if transformer_path.is_file():
    try:
        from transformer import load_transformer

        transformer_model = load_transformer(str(transformer_path))
        print(f"SWIN Transformer loaded: {transformer_path}")
    except Exception as error:
        print(f"SWIN Transformer load error: {error}")

if rnn_path.is_file():
    try:
        rnn_model = TinyCharRNN.load(str(rnn_path))
        print(f"SWIN RNN loaded: {rnn_path}")
    except Exception as error:
        print(f"SWIN RNN load error: {error}")

# In-memory session history: isolated by session_id and limited to the latest turns.
conversation_memory: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
MAX_SESSIONS = 256
MAX_TURNS_PER_SESSION = 6


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="android-default", min_length=1, max_length=80)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


def remember_turn(session_id: str, message: str, answer: str) -> int:
    turns = conversation_memory.setdefault(session_id, [])
    turns.append((message, answer))
    del turns[:-MAX_TURNS_PER_SESSION]
    conversation_memory.move_to_end(session_id)
    while len(conversation_memory) > MAX_SESSIONS:
        conversation_memory.popitem(last=False)
    return len(turns)


@app.get("/")
def root():
    return {
        "name": "SWIN AI",
        "status": "online",
        "transformer_loaded": transformer_model is not None,
        "rnn_loaded": rnn_model is not None,
        "knowledge_entries": knowledge.size,
        "active_sessions": len(conversation_memory),
        "version": app.version,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "transformer_loaded": transformer_model is not None,
        "rnn_loaded": rnn_model is not None,
        "knowledge_entries": knowledge.size,
    }


@app.post("/memory/clear")
def clear_memory(request: ClearMemoryRequest):
    conversation_memory.pop(request.session_id, None)
    return {"ok": True, "session_id": request.session_id, "memory_turns": 0}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    session_id = request.session_id.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Сообщение не должно быть пустым.")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id не должен быть пустым.")

    history = conversation_memory.get(session_id, [])

    # Retrieval has priority for known technical facts; generation is used for new questions.
    match = knowledge.search(message)
    if match is not None:
        answer = match["answer"]
        turns = remember_turn(session_id, message, answer)
        return {
            "answer": answer,
            "mode": "local_knowledge",
            "source": match["title"],
            "confidence": round(match["score"], 3),
            "memory_turns": turns,
        }

    if transformer_model is not None:
        prompt_parts = []
        for old_user, old_answer in history[-MAX_TURNS_PER_SESSION:]:
            prompt_parts.append(f"Пользователь: {old_user}\nSWIN: {old_answer}")
        prompt_parts.append(f"Пользователь: {message}\nSWIN:")
        prompt = "\n".join(prompt_parts)
        try:
            answer = transformer_model.generate(
                prompt,
                max_new_tokens=280,
                temperature=0.72,
                top_k=40,
                stop_at_newline=True,
            ).strip()
            answer = answer.split("\nПользователь:", 1)[0].strip()
            if not answer:
                answer = "Пока не смог сформировать связный ответ. Попробуй уточнить вопрос."
            turns = remember_turn(session_id, message, answer)
            return {
                "answer": answer,
                "mode": "byte_transformer",
                "source": "собственные веса SWIN Transformer",
                "confidence": None,
                "memory_turns": turns,
            }
        except Exception as error:
            print(f"Transformer generation error: {error}")

    if rnn_model is not None:
        prompt = f"Пользователь: {message}\nSWIN:"
        generated = rnn_model.generate(prompt, length=220, temperature=0.75)
        answer = generated[len(prompt):].split("\n", 1)[0].strip()
        if not answer:
            answer = "Пока не смог продолжить текст. Добавь больше проверенных примеров для обучения."
        turns = remember_turn(session_id, message, answer)
        return {
            "answer": answer,
            "mode": "character_rnn",
            "source": "собственные веса SWIN RNN",
            "confidence": None,
            "memory_turns": turns,
        }

    raise HTTPException(
        status_code=503,
        detail=(
            "В базе знаний нет подходящего ответа, а обученные веса Transformer/RNN "
            "не загружены. Добавь checkpoint модели на сервер."
        ),
    )
