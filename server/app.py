import os
import re
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from auth import AuthStore, TOKEN_TTL_SECONDS, validate_email
from knowledge import KnowledgeBase
from model import TinyCharRNN

app = FastAPI(title="SWIN AI", version="0.5.0")

database_path = Path(os.getenv("SWIN_DATABASE_PATH", "data/swin.sqlite3"))
auth_store = AuthStore(database_path)
bearer_scheme = HTTPBearer(auto_error=False)

rnn_model = None
transformer_model = None
rnn_path = Path(os.getenv("SWIN_MODEL_PATH", "model.npz"))
transformer_path = Path(os.getenv("SWIN_TRANSFORMER_PATH", "models/swin-transformer.pt"))
knowledge_path = Path(os.getenv("SWIN_KNOWLEDGE_PATH", "data/knowledge.json"))
knowledge = KnowledgeBase(knowledge_path)

# Keep the CPU thread pool small on modest Render instances.
try:
    import torch

    cpu_threads = max(1, min(4, int(os.getenv("SWIN_NUM_THREADS", "2"))))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except ImportError:
    pass

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

# Conversation history is isolated by authenticated user and client session.
conversation_memory: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
MAX_SESSIONS = 256
MAX_TURNS_PER_SESSION = 6

_auth_attempts: dict[str, deque[float]] = defaultdict(deque)
_auth_attempts_lock = Lock()
AUTH_WINDOW_SECONDS = 300


class AuthRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(AuthRequest):
    password: str = Field(min_length=12, max_length=128)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=80)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


def enforce_auth_rate_limit(request: Request, action: str, max_attempts: int) -> None:
    address = request.client.host if request.client else "unknown"
    key = f"{action}:{address}"
    now = time.monotonic()
    with _auth_attempts_lock:
        attempts = _auth_attempts[key]
        while attempts and now - attempts[0] > AUTH_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= max_attempts:
            raise HTTPException(
                status_code=429,
                detail="Слишком много попыток. Подожди несколько минут.",
                headers={"Retry-After": str(AUTH_WINDOW_SECONDS)},
            )
        attempts.append(now)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Войди в SWIN AI, чтобы продолжить.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = auth_store.user_for_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Сессия истекла или недействительна. Войди снова.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


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
        "auth": "email-password",
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


@app.post("/auth/register")
def register(request: RegisterRequest, http_request: Request):
    enforce_auth_rate_limit(http_request, "register", 5)
    try:
        email = validate_email(request.email)
        user = auth_store.create_user(email, request.password)
    except ValueError as error:
        message = str(error)
        status = 409 if "уже существует" in message else 400
        raise HTTPException(status_code=status, detail=message) from error

    token, expires_at = auth_store.issue_token(user["id"])
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_TTL_SECONDS,
        "expires_at": expires_at,
        "user": user,
    }


@app.post("/auth/login")
def login(request: AuthRequest, http_request: Request):
    enforce_auth_rate_limit(http_request, "login", 10)
    user = auth_store.authenticate(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Неверная почта или пароль.")
    token, expires_at = auth_store.issue_token(user["id"])
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_TTL_SECONDS,
        "expires_at": expires_at,
        "user": user,
    }


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"user": user}


@app.post("/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    user: dict = Depends(get_current_user),
):
    auth_store.revoke_token(credentials.credentials)
    return {"ok": True, "email": user["email"]}


@app.post("/memory/clear")
def clear_memory(
    request: ClearMemoryRequest,
    user: dict = Depends(get_current_user),
):
    memory_key = f"{user['id']}:{request.session_id}"
    conversation_memory.pop(memory_key, None)
    return {"ok": True, "session_id": request.session_id, "memory_turns": 0}


@app.post("/chat")
def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    started_at = time.perf_counter()
    message = request.message.strip()
    session_id = request.session_id.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Сообщение не должно быть пустым.")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id не должен быть пустым.")

    memory_key = f"{user['id']}:{session_id}"
    history = conversation_memory.get(memory_key, [])

    # Retrieval returns known verified facts without invoking token-by-token generation.
    match = knowledge.search(message)
    if match is not None:
        answer = match["answer"]
        turns = remember_turn(memory_key, message, answer)
        return {
            "answer": answer,
            "mode": "local_knowledge",
            "source": match["title"],
            "confidence": round(match["score"], 3),
            "memory_turns": turns,
            "generation_ms": round((time.perf_counter() - started_at) * 1000, 1),
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
                max_new_tokens=160,
                temperature=0.0,
                top_k=1,
                stop_at_newline=True,
            ).strip()
            answer = answer.split("\nПользователь:", 1)[0].strip()
            if not answer:
                answer = "Пока не смог сформировать связный ответ. Попробуй уточнить вопрос."
            turns = remember_turn(memory_key, message, answer)
            elapsed = max(time.perf_counter() - started_at, 0.001)
            try:
                from transformer import ByteTokenizer

                generated_bytes = len(ByteTokenizer.encode(answer))
            except Exception:
                generated_bytes = len(answer.encode("utf-8"))
            return {
                "answer": answer,
                "mode": "byte_transformer",
                "source": "собственные веса SWIN Transformer",
                "confidence": None,
                "memory_turns": turns,
                "generation_ms": round(elapsed * 1000, 1),
                "generation_bytes_per_second": round(generated_bytes / elapsed, 1),
            }
        except Exception as error:
            print(f"Transformer generation error: {error}")

    if rnn_model is not None:
        prompt = f"Пользователь: {message}\nSWIN:"
        generated = rnn_model.generate(prompt, length=220, temperature=0.75)
        answer = generated[len(prompt):].split("\n", 1)[0].strip()
        if not answer:
            answer = "Пока не смог продолжить текст. Добавь больше проверенных примеров для обучения."
        turns = remember_turn(memory_key, message, answer)
        return {
            "answer": answer,
            "mode": "character_rnn",
            "source": "собственные веса SWIN RNN",
            "confidence": None,
            "memory_turns": turns,
            "generation_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }

    raise HTTPException(
        status_code=503,
        detail=(
            "В базе знаний нет подходящего ответа, а обученные веса Transformer/RNN "
            "не загружены. Добавь checkpoint модели на сервер."
        ),
    )
