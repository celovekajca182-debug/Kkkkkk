import os
import re
import time
import urllib.request
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from auth import AuthStore, TOKEN_TTL_SECONDS, validate_email
from chat_history import ChatHistoryStore
from coding_agent import LANGUAGES as CODE_LANGUAGES, run_code_assist
from language_support import (
    SUPPORTED_CHAT_LANGUAGES,
    format_chat_prompt,
    localized_fallback,
    localized_small_talk,
    resolve_response_language,
)
from postgres_stores import (
    PostgresAuthStore,
    PostgresChatHistoryStore,
    migrate_sqlite_to_postgres,
)
from knowledge import KnowledgeBase
from model import TinyCharRNN

app = FastAPI(title="SWIN AI", version="0.6.0")

database_path = Path(os.getenv("SWIN_DATABASE_PATH", "data/swin.sqlite3"))
database_url = os.getenv("DATABASE_URL", "").strip()
if database_url:
    auth_store = PostgresAuthStore(database_url)
    chat_history = PostgresChatHistoryStore(database_url)
    migration_result = migrate_sqlite_to_postgres(database_path, database_url)
    print(
        "SWIN PostgreSQL stores enabled; legacy SQLite rows copied: "
        f"users={migration_result['users']}, "
        f"tokens={migration_result['tokens']}, "
        f"messages={migration_result['messages']}"
    )
else:
    auth_store = AuthStore(database_path)
    chat_history = ChatHistoryStore(database_path)
    print("SWIN SQLite stores enabled; set DATABASE_URL for durable cross-device memory.")
bearer_scheme = HTTPBearer(auto_error=False)

rnn_model = None
transformer_model = None
rnn_path = Path(os.getenv("SWIN_MODEL_PATH", "model.npz"))
transformer_path = Path(os.getenv("SWIN_TRANSFORMER_PATH", "models/swin-transformer.pt"))
transformer_url = os.getenv(
    "SWIN_TRANSFORMER_URL",
    "https://github.com/celovekajca182-debug/Kkkkkk/releases/download/swin-model-latest/swin-transformer.pt",
)

# Render's filesystem is ephemeral: restore the public checkpoint at each cold start.
if not transformer_path.is_file() and transformer_url:
    try:
        transformer_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = transformer_path.with_suffix(transformer_path.suffix + ".download")
        request = urllib.request.Request(
            transformer_url,
            headers={"User-Agent": "SWIN-AI-Model-Loader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response, temporary_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary_path.stat().st_size < 1024:
            raise ValueError("Скачанный checkpoint слишком мал.")
        temporary_path.replace(transformer_path)
        print(f"SWIN Transformer checkpoint downloaded: {transformer_path}")
    except Exception as error:
        print(f"SWIN Transformer download error: {error}")
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass
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
    response_language: str = Field(default="auto", min_length=2, max_length=16)


class CodeAssistRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="auto", min_length=1, max_length=32)
    research_web: bool = True
    show_activity: bool = False
    project_context: str = Field(default="", max_length=4000)
    session_id: str = Field(default="", max_length=80)


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=120000)


class SyncMemoryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    messages: list[HistoryMessage] = Field(default_factory=list, max_length=80)


def record_chat_turn(
    memory_key: str,
    user_id: int,
    session_id: str,
    message: str,
    answer: str,
) -> int:
    """Keep a bounded hot cache and persist the complete turn in SQLite."""
    count = remember_turn(memory_key, message, answer)
    chat_history.save_turn(user_id, session_id, message, answer)
    return count


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


def safe_small_talk(message: str) -> str | None:
    """Return deterministic Russian replies for common greetings and small talk."""
    normalized = re.sub(r"[^a-zа-яё0-9 ]", " ", message.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if re.fullmatch(r"(привет|здравствуй|здравствуйте|добрый день|доброе утро|добрый вечер|хай|салют)", normalized):
        return "Привет! Я на связи. Напиши, что нужно сделать или объяснить."
    if re.fullmatch(r"(как дела|как ты|как поживаешь|ты как)", normalized):
        return "Всё хорошо. Я на связи и готов помочь. Что будем делать?"
    if re.fullmatch(r"(спасибо|спс|благодарю)", normalized):
        return "Пожалуйста! Если понадобится, продолжим."
    if re.fullmatch(r"(пока|до свидания|до встречи)", normalized):
        return "Пока! Возвращайся, когда понадобится помощь."
    return None


def looks_like_garbage(text: str) -> bool:
    """Reject malformed and structurally implausible natural-language output."""
    value = text.strip()
    if len(value) < 3 or "\ufffd" in value:
        return True

    letters = [ch.lower() for ch in value if ch.isalpha()]
    if len(letters) >= 8:
        compact = "".join(letters)
        if re.search(r"(.)\1{5,}", compact):
            return True
        if len(set(letters)) / len(letters) < 0.18:
            return True

    has_cjk = any(
        "\u4e00" <= char <= "\u9fff"
        or "\u3040" <= char <= "\u30ff"
        or "\uac00" <= char <= "\ud7af"
        for char in value
    )
    words = re.findall(r"[a-zа-яё]{2,}", value.lower())
    if len(letters) >= 16 and len(words) < 2 and not has_cjk:
        return True

    # A chat answer should not be a long unbroken token or a repeated vowel stream.
    if len(letters) >= 24:
        if re.search(r"[аеёиоуыэюя]{4,}", value.lower()):
            return True
        trigrams = [
            "".join(letters[index:index + 3])
            for index in range(len(letters) - 2)
        ]
        if len(trigrams) >= 20:
            counts = Counter(trigrams)
            diversity = len(counts) / len(trigrams)
            dominant = counts.most_common(1)[0][1] / len(trigrams)
            if diversity < 0.36 or dominant > 0.16:
                return True
        if not has_cjk:
            vowel_count = sum(char in "аеёиоуыэюяaeiou" for char in letters)
            vowel_ratio = vowel_count / len(letters)
            if vowel_ratio > 0.68 or vowel_ratio < 0.12:
                return True

    return False


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
        "storage_backend": "postgresql" if database_url else "sqlite",
        "persistent_storage": bool(database_url),
        "cross_device_memory": bool(database_url),
        "code_quality_guard": True,
        "version": app.version,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "transformer_loaded": transformer_model is not None,
        "rnn_loaded": rnn_model is not None,
        "knowledge_entries": knowledge.size,
        "storage_backend": "postgresql" if database_url else "sqlite",
        "persistent_storage": bool(database_url),
        "cross_device_memory": bool(database_url),
        "code_quality_guard": True,
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
    deleted_messages = chat_history.clear(int(user["id"]), request.session_id)
    return {
        "ok": True,
        "session_id": request.session_id,
        "memory_turns": 0,
        "deleted_messages": deleted_messages,
    }


@app.get("/memory/history")
def get_memory_history(
    session_id: str = Query(min_length=1, max_length=80),
    user: dict = Depends(get_current_user),
):
    """Return only the authenticated user's messages for the requested session."""
    messages = chat_history.messages(int(user["id"]), session_id, limit=80)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@app.post("/memory/sync")
def sync_memory(
    request: SyncMemoryRequest,
    user: dict = Depends(get_current_user),
):
    """Restore a device backup only if no server-side history exists for this session."""
    restored = chat_history.restore_messages(
        int(user["id"]),
        request.session_id,
        [item.model_dump() for item in request.messages],
    )
    messages = chat_history.messages(int(user["id"]), request.session_id, limit=80)
    return {
        "ok": True,
        "restored": restored,
        "count": len(messages),
        "session_id": request.session_id,
    }


@app.post("/code/assist")
def code_assist(
    request: CodeAssistRequest,
    user: dict = Depends(get_current_user),
):
    """Research a coding task and return a generated, statically checked draft."""
    started_at = time.perf_counter()
    session_id = request.session_id.strip()
    recent_turns = (
        chat_history.turns(int(user["id"]), session_id, limit=2)
        if session_id else []
    )

    # Carry a compact, account-scoped project context into coding requests.
    # The model has a short context window, so recent decisions are summarized
    # rather than blindly appending entire previous code files.
    context_parts: list[str] = []
    supplied_context = request.project_context.strip()
    if supplied_context:
        context_parts.append("Контекст пользователя: " + supplied_context[:150])
    if recent_turns:
        memory_lines = []
        for previous_task, previous_answer in recent_turns[-2:]:
            answer_summary = previous_answer.split("[SWIN_CODE_BLOCK]", 1)[0].strip()
            memory_lines.append(
                f"Запрос: {previous_task[:70]} | Итог: {answer_summary[:90]}"
            )
        context_parts.append("ПАМЯТЬ СЕССИИ: " + " || ".join(memory_lines))
    project_context = "\n".join(context_parts)[:4000]

    try:
        result = run_code_assist(
            task=request.task,
            language=request.language,
            transformer_model=transformer_model,
            research_web=request.research_web,
            project_context=project_context,
            show_activity=request.show_activity,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    result["model_loaded"] = transformer_model is not None
    result["memory_context_turns"] = len(recent_turns)
    if session_id:
        assistant_text = result.get("summary", "Результат режима разработки.")
        generated_code = result.get("code", "")
        if generated_code:
            assistant_text += "\n\n[SWIN_CODE_BLOCK]\n" + generated_code
        chat_history.save_turn(
            int(user["id"]),
            session_id,
            request.task.strip(),
            assistant_text[:120000],
        )
    return result


@app.post("/chat")
def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    started_at = time.perf_counter()
    message = request.message.strip()
    session_id = request.session_id.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Сообщение не должно быть пустым.")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id не должен быть пустым.")

    response_language = resolve_response_language(message, request.response_language)
    memory_key = f"{user['id']}:{session_id}"
    history = chat_history.turns(int(user["id"]), session_id, limit=MAX_TURNS_PER_SESSION)
    if not history:
        history = conversation_memory.get(memory_key, [])

    small_talk = localized_small_talk(message, response_language)
    if small_talk is None and response_language == "ru":
        small_talk = safe_small_talk(message)
    if small_talk is not None:
        turns = record_chat_turn(
            memory_key, int(user["id"]), session_id, message, small_talk
        )
        return {
            "answer": small_talk,
            "mode": "safe_small_talk",
            "source": "встроенный безопасный ответ",
            "confidence": 1.0,
            "response_language": response_language,
            "memory_turns": turns,
            "generation_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }

    # The current curated KnowledgeBase is Russian-only. Never return its Russian
    # answer when the user selected a different response language.
    if response_language == "ru":
        match = knowledge.search(message)
        if match is not None:
            answer = match["answer"]
            turns = record_chat_turn(memory_key, int(user["id"]), session_id, message, answer)
            return {
                "answer": answer,
                "mode": "local_knowledge",
                "source": match["title"],
                "confidence": round(match["score"], 3),
                "response_language": response_language,
                "memory_turns": turns,
                "generation_ms": round((time.perf_counter() - started_at) * 1000, 1),
            }

    if transformer_model is not None:
        prompt = format_chat_prompt(history, message, response_language)
        try:
            generated_answer = transformer_model.generate(
                prompt,
                max_new_tokens=200,
                temperature=0.0,
                top_k=1,
                stop_at_newline=False,
            ).strip()
            generated_answer = generated_answer.split("\nUser:", 1)[0].strip()
            generated_answer = generated_answer.split("\nПользователь:", 1)[0].strip()
            generation_valid = bool(generated_answer) and not looks_like_garbage(generated_answer)
            if generation_valid:
                answer = generated_answer
                tokenizer = getattr(transformer_model, "tokenizer", None)
                tokenizer_type = getattr(tokenizer, "tokenizer_type", None)
                if tokenizer_type == "byte_bpe":
                    response_mode = "bpe_byte_transformer"
                    response_source = "собственные веса SWIN Transformer (Byte-level BPE)"
                elif getattr(tokenizer, "vocabulary", None) is not None:
                    response_mode = "unicode_char_transformer"
                    response_source = "собственные веса SWIN Transformer (Unicode)"
                else:
                    response_mode = "byte_transformer"
                    response_source = "собственные веса SWIN Transformer (UTF-8 bytes)"
            else:
                answer = localized_fallback(response_language)
                response_mode = "safe_fallback"
                response_source = "защита от бессвязной генерации"
            turns = record_chat_turn(memory_key, int(user["id"]), session_id, message, answer)
            elapsed = max(time.perf_counter() - started_at, 0.001)
            response = {
                "answer": answer,
                "mode": response_mode,
                "source": response_source,
                "confidence": None,
                "response_language": response_language,
                "memory_turns": turns,
                "generation_ms": round(elapsed * 1000, 1),
            }
            if generation_valid:
                tokenizer = getattr(transformer_model, "tokenizer", None)
                try:
                    generated_tokens = (
                        len(tokenizer.encode(generated_answer))
                        if tokenizer is not None
                        else len(generated_answer.encode("utf-8"))
                    )
                except Exception:
                    generated_tokens = len(generated_answer.encode("utf-8"))
                response["generation_tokens_per_second"] = round(generated_tokens / elapsed, 1)
                response["tokenizer"] = (
                    getattr(tokenizer, "tokenizer_type", None)
                    or ("unicode_char" if getattr(tokenizer, "vocabulary", None) is not None else "utf8_bytes")
                )
            return response
        except Exception as error:
            print(f"Transformer generation error: {error}")

    # The legacy RNN was trained only in Russian; never use it for other languages.
    if rnn_model is not None and response_language == "ru":
        prompt = f"Пользователь: {message}\nSWIN:"
        try:
            generated = rnn_model.generate(prompt, length=220, temperature=0.75)
            generated_answer = generated[len(prompt):].split("\n", 1)[0].strip()
        except Exception as error:
            print(f"RNN generation error: {error}")
            generated_answer = ""
        if generated_answer and not looks_like_garbage(generated_answer):
            answer = generated_answer
            response_mode = "character_rnn"
            response_source = "собственные веса SWIN RNN"
        else:
            answer = localized_fallback(response_language)
            response_mode = "safe_fallback"
            response_source = "защита от бессвязной генерации"
        turns = record_chat_turn(memory_key, int(user["id"]), session_id, message, answer)
        return {
            "answer": answer,
            "mode": response_mode,
            "source": response_source,
            "confidence": None,
            "response_language": response_language,
            "memory_turns": turns,
            "generation_ms": round((time.perf_counter() - started_at) * 1000, 1),
        }

    raise HTTPException(status_code=503, detail=localized_fallback(response_language))


@app.get("/capabilities")
def capabilities():
    return {
        "chat_languages": [
            {"code": code, "label": label}
            for code, label in SUPPORTED_CHAT_LANGUAGES.items()
        ],
        "default_chat_language": "auto",
        "auto_detection": "heuristic",
        "code_languages": [
            {"code": code, "label": label}
            for code, label in CODE_LANGUAGES.items()
        ],
        "code_validation": {
            "execution": False,
            "python": "AST parse only; program is not executed",
            "json_xml_toml": "standard-library parser",
            "other": "language-dependent static checks or delimiter scan",
        },
    }
