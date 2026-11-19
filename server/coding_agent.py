"""Research-first coding assistant for SWIN AI.

Searches the broad public web, ranks primary sources above community material,
and performs safe static checks. Generated code is never executed on the API server.
"""
from __future__ import annotations

import ast
import html.parser
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlparse
from language_support import code_assist_copy, response_language_instruction, resolve_response_language

LANGUAGES = {
    "auto": "Автоопределение",
    "python": "Python",
    "gdscript": "Godot 4 / GDScript",
    "kotlin": "Kotlin",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "c": "C",
    "cpp": "C++",
    "csharp": "C#",
    "go": "Go",
    "rust": "Rust",
    "php": "PHP",
    "ruby": "Ruby",
    "swift": "Swift",
    "dart": "Dart",
    "sql": "SQL",
    "bash": "Bash / Shell",
    "powershell": "PowerShell",
    "lua": "Lua",
    "r": "R",
    "scala": "Scala",
    "perl": "Perl",
    "elixir": "Elixir",
    "haskell": "Haskell",
    "html": "HTML",
    "css": "CSS",
    "xml": "XML",
    "json": "JSON",
    "yaml": "YAML",
    "toml": "TOML",
    "dockerfile": "Dockerfile",
    "glsl": "GLSL / Shaders",
    "solidity": "Solidity",
    "other": "Другой язык / свой формат",
}

LANGUAGE_ALIASES = {
    "c++": "cpp", "c#": "csharp", "sh": "bash", "shell": "bash",
    "zsh": "bash", "ps1": "powershell", "gdscript 4": "gdscript",
    "godot": "gdscript", "js": "javascript", "ts": "typescript",
    "golang": "go", "rs": "rust", "py": "python", "kt": "kotlin",
    "cs": "csharp", "yml": "yaml", "md": "other",
}

LANGUAGE_HINTS = {
    "gdscript": ("godot", "gdscript", "гдскрипт"),
    "python": ("python", "питон", "fastapi", "django", "flask"),
    "kotlin": ("kotlin", "котлин"),
    "java": ("java", "джавa"),
    "javascript": ("javascript", "node.js", "nodejs", "react", "vue", "js "),
    "typescript": ("typescript", " ts ", "next.js"),
    "cpp": ("c++", "cpp", "c plus plus"),
    "csharp": ("c#", "c sharp", ".net", "unity"),
    "c": ("язык c ", "системный c", "gcc c "),
    "rust": ("rust", "rustlang", "раст"),
    "go": ("golang", "язык go", " go "),
    "php": ("php", "laravel"),
    "ruby": ("ruby", "rails"),
    "swift": ("swift", "ios", "swiftui"),
    "dart": ("dart", "flutter"),
    "sql": ("sql", "postgres", "sqlite", "mysql"),
    "bash": ("bash", "shell script", "термукс", "termux"),
    "powershell": ("powershell", "pwsh"),
    "lua": ("lua", "roblox lua"),
    "r": ("язык r", " r language", "r статист"),
    "scala": ("scala",),
    "perl": ("perl",),
    "elixir": ("elixir",),
    "haskell": ("haskell",),
    "html": ("html", "веб-страниц", "разметк"),
    "css": ("css", "стил", "верстк"),
    "solidity": ("solidity", "smart contract", "смарт-контракт"),
    "glsl": ("glsl", "shader", "шейдер"),
}

# These domains are ranked as primary references. Search itself is not restricted
# to them: Stack Overflow, GitHub, forums, tutorials and other indexed sites remain
# discoverable and are labelled as secondary/community material.
PRIMARY_DOC_DOMAINS = {
    "python": ("docs.python.org",),
    "gdscript": ("docs.godotengine.org",),
    "kotlin": ("kotlinlang.org", "developer.android.com"),
    "java": ("docs.oracle.com", "dev.java"),
    "javascript": ("developer.mozilla.org", "nodejs.org", "tc39.es"),
    "typescript": ("typescriptlang.org", "developer.mozilla.org"),
    "c": ("en.cppreference.com", "gcc.gnu.org", "clang.llvm.org"),
    "cpp": ("en.cppreference.com", "isocpp.org", "gcc.gnu.org", "clang.llvm.org"),
    "csharp": ("learn.microsoft.com", "dotnet.microsoft.com"),
    "go": ("go.dev", "pkg.go.dev"),
    "rust": ("doc.rust-lang.org", "rust-lang.org"),
    "php": ("php.net",),
    "ruby": ("ruby-lang.org", "docs.ruby-lang.org"),
    "swift": ("swift.org", "developer.apple.com"),
    "dart": ("dart.dev", "docs.flutter.dev"),
    "sql": ("sqlite.org", "postgresql.org", "dev.mysql.com"),
    "bash": ("gnu.org", "man7.org"),
    "powershell": ("learn.microsoft.com",),
    "lua": ("lua.org",),
    "r": ("cran.r-project.org", "r-project.org"),
    "scala": ("scala-lang.org",),
    "perl": ("perldoc.perl.org", "perl.org"),
    "elixir": ("elixir-lang.org",),
    "haskell": ("haskell.org", "downloads.haskell.org"),
    "html": ("html.spec.whatwg.org", "developer.mozilla.org"),
    "css": ("w3.org", "developer.mozilla.org"),
    "xml": ("w3.org", "developer.mozilla.org"),
    "json": ("json.org", "ecma-international.org"),
    "yaml": ("yaml.org",),
    "toml": ("toml.io",),
    "dockerfile": ("docs.docker.com",),
    "glsl": ("registry.khronos.org", "wikis.khronos.org"),
    "solidity": ("docs.soliditylang.org",),
}

COMMUNITY_DOMAINS = (
    "stackoverflow.com", "stackexchange.com", "github.com", "gitlab.com",
    "reddit.com", "dev.to", "medium.com", "hashnode.com",
)


def normalize_language(requested: str) -> str:
    value = (requested or "auto").strip().lower()
    value = LANGUAGE_ALIASES.get(value, value)
    return value if value in LANGUAGES else "other"


def detect_language(task: str, requested: str) -> str:
    normalized = normalize_language(requested)
    if normalized != "auto":
        return normalized
    lowered = " " + (task or "").lower() + " "
    for language, hints in LANGUAGE_HINTS.items():
        if any(hint in lowered for hint in hints):
            return language
    return "python"


def _domain_matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _safe_search_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not host or parsed.username or parsed.password:
            return False
        # Literal IPs (including loopback/private/link-local) are never opened.
        try:
            address = ipaddress.ip_address(host)
            return address.is_global
        except ValueError:
            pass
        if host in {"localhost", "localhost.localdomain"} or "." not in host:
            return False
        return True
    except (TypeError, ValueError):
        return False


def _source_quality(url: str, language: str) -> tuple[int, str]:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    primary = PRIMARY_DOC_DOMAINS.get(language, ())
    if _domain_matches(host, primary):
        return 3, "Официальная документация"
    if _domain_matches(host, COMMUNITY_DOMAINS):
        return 2, "Сообщество разработчиков"
    return 1, "Публичный веб-источник"


def _search_query(task: str, language: str) -> str:
    compact_task = re.sub(r"\s+", " ", task).strip()[:260]
    label = LANGUAGES.get(language, language)
    return (
        f"{compact_task} {label} implementation API reference examples "
        "known issues compatibility best practices"
    ).strip()


def search_documentation(
    task: str, language: str, max_results: int = 7
) -> list[dict[str, str]]:
    """Search the broad public web; extract full text only from primary doc hosts."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    try:
        client = DDGS(timeout=8)
        raw_results = client.text(
            _search_query(task, language),
            region="wt-wt",
            safesearch="moderate",
            max_results=max(1, min(max_results * 3, 21)),
            backend="auto",
        )
    except Exception:
        return []

    ranked: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = set()
    for position, item in enumerate(raw_results or []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("href", "")).strip()
        if not _safe_search_url(url):
            continue
        parsed = urlparse(url)
        canonical = (parsed.hostname or "").lower() + parsed.path.rstrip("/")
        if canonical in seen:
            continue
        seen.add(canonical)
        score, source_type = _source_quality(url, language)
        snippet = re.sub(r"\s+", " ", str(item.get("body", ""))).strip()[:650]
        title = re.sub(r"\s+", " ", str(item.get("title", "Источник из интернета"))).strip()
        # Only request full-page extraction from allow-listed primary references.
        if score == 3 and len(snippet) < 180 and len(ranked) < 2:
            try:
                extracted = client.extract(url, fmt="text_markdown")
                if isinstance(extracted, dict):
                    page_text = re.sub(
                        r"\s+", " ", str(extracted.get("content", ""))
                    ).strip()
                    if page_text:
                        snippet = page_text[:1100]
            except Exception:
                pass
        ranked.append((score, -position, {
            "title": title[:180] or "Источник из интернета",
            "url": url[:700],
            "snippet": snippet or "Поисковик нашёл страницу; подробный фрагмент недоступен.",
            "source_type": source_type,
            "trust": "primary" if score == 3 else ("community" if score == 2 else "unverified"),
        }))

    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [entry[2] for entry in ranked[:max(1, min(max_results, 10))]]


def looks_like_garbage_code(code: str) -> bool:
    """Reject obvious byte-level decoding failures before returning a code draft."""
    value = (code or "").strip()
    if not value or "\ufffd" in value:
        return True
    letters = [char.lower() for char in value if char.isalpha()]
    if len(letters) >= 10:
        diversity = len(set(letters)) / len(letters)
        dominant = max(letters.count(char) for char in set(letters)) / len(letters)
        if diversity < 0.12 or dominant > 0.52:
            return True
        if re.search(r"(.)\1{5,}", "".join(letters)):
            return True
    return False


def _extract_code(text: str) -> str:
    fence = chr(96) * 3
    pattern = re.escape(fence) + r"(?:[a-zA-Z0-9_+.#-]+)?\s*\n(.*?)" + re.escape(fence)
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().strip(chr(96)).strip()


def _run_static_tool(command: list[str], code: str, suffix: str) -> dict[str, Any] | None:
    """Run only compiler/parser check modes, never execute the generated program."""
    if not shutil.which(command[0]):
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="swin-check-") as directory:
            path = os.path.join(directory, "Main" + suffix)
            with open(path, "w", encoding="utf-8", newline="\n") as target:
                target.write(code)
            args = [part.replace("{file}", path).replace("{dir}", directory) for part in command]
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=8,
                cwd=directory,
                check=False,
            )
        if completed.returncode == 0:
            return {
                "status": "passed",
                "method": command[0] + "_static",
                "message": "Статический анализатор " + command[0] + " не обнаружил синтаксических ошибок.",
            }
        detail = (completed.stderr or completed.stdout or "Проверка завершилась с ошибкой.")
        detail = re.sub(r"\s+", " ", detail).strip()[:260]
        return {
            "status": "failed",
            "method": command[0] + "_static",
            "message": detail,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "limited",
            "method": command[0] + "_static",
            "message": "Статическая проверка превысила лимит 8 секунд.",
        }
    except (OSError, UnicodeError) as error:
        return {
            "status": "limited",
            "method": command[0] + "_static",
            "message": "Не удалось запустить статический анализатор: " + str(error)[:180],
        }


def _check_delimiters(code: str, language: str) -> dict[str, Any]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(code):
        char = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if char == "\n":
            line_comment = False
            i += 1
            continue
        if line_comment:
            i += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        elif char == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        elif char == "#" and language in {"gdscript", "python", "bash", "ruby", "yaml", "r", "powershell"}:
            line_comment = True
        elif char in pairs:
            stack.append(char)
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return {"status": "failed", "method": "delimiter_scan", "message": "Не совпадают скобки."}
        i += 1
    if quote or block_comment or stack:
        return {
            "status": "failed",
            "method": "delimiter_scan",
            "message": "Возможно, не закрыта строка, комментарий или скобка.",
        }
    return {
        "status": "limited",
        "method": "delimiter_scan",
        "message": "Баланс строк, комментариев и скобок проверен; полноценный компилятор для этого языка не установлен.",
    }


class _HTMLCheck(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.failed = False

    def unknown_decl(self, data: str) -> None:
        self.failed = True


def validate_code(code: str, language: str) -> dict[str, Any]:
    """Validate syntax without running user code or project entry points."""
    if not code or not code.strip():
        return {"status": "failed", "method": "empty", "message": "Модель не вернула код."}
    if len(code) > 100_000:
        return {"status": "limited", "method": "size_guard", "message": "Файл больше лимита безопасной статической проверки."}

    language = normalize_language(language)
    if language == "python":
        try:
            ast.parse(code)
            return {"status": "passed", "method": "python_ast", "message": "Синтаксис Python корректен (AST). Код не запускался."}
        except SyntaxError as error:
            return {"status": "failed", "method": "python_ast", "message": f"Ошибка Python в строке {error.lineno}: {error.msg}"}
    if language == "json":
        try:
            json.loads(code)
            return {"status": "passed", "method": "json_parser", "message": "JSON корректен."}
        except json.JSONDecodeError as error:
            return {"status": "failed", "method": "json_parser", "message": f"Ошибка JSON в строке {error.lineno}: {error.msg}"}
    if language == "xml":
        try:
            ET.fromstring(code)
            return {"status": "passed", "method": "xml_parser", "message": "XML корректен."}
        except ET.ParseError as error:
            return {"status": "failed", "method": "xml_parser", "message": f"Ошибка XML: {error}"}
    if language == "toml":
        try:
            tomllib.loads(code)
            return {"status": "passed", "method": "tomllib", "message": "TOML корректен."}
        except tomllib.TOMLDecodeError as error:
            return {"status": "failed", "method": "tomllib", "message": f"Ошибка TOML: {error}"}
    if language == "javascript":
        return _run_static_tool(["node", "--check", "{file}"], code, ".mjs") or _check_delimiters(code, language)
    if language == "typescript":
        return _run_static_tool(["tsc", "--noEmit", "--skipLibCheck", "--target", "ES2020", "{file}"], code, ".ts") or _check_delimiters(code, language)
    if language == "bash":
        return _run_static_tool(["bash", "-n", "{file}"], code, ".sh") or _check_delimiters(code, language)
    if language == "ruby":
        return _run_static_tool(["ruby", "-c", "{file}"], code, ".rb") or _check_delimiters(code, language)
    if language == "php":
        return _run_static_tool(["php", "-l", "{file}"], code, ".php") or _check_delimiters(code, language)
    if language == "go":
        return _run_static_tool(["gofmt", "-d", "{file}"], code, ".go") or _check_delimiters(code, language)
    if language == "swift":
        return _run_static_tool(["swiftc", "-parse", "{file}"], code, ".swift") or _check_delimiters(code, language)
    if language == "c":
        return _run_static_tool(["cc", "-fsyntax-only", "-x", "c", "{file}"], code, ".c") or _check_delimiters(code, language)
    if language == "cpp":
        return _run_static_tool(["c++", "-fsyntax-only", "-x", "c++", "{file}"], code, ".cpp") or _check_delimiters(code, language)
    if language == "java":
        return _run_static_tool(["javac", "-proc:none", "-d", "{dir}", "{file}"], code, ".java") or _check_delimiters(code, language)
    if language == "rust":
        return _run_static_tool(["rustc", "--emit=metadata", "--crate-type", "lib", "-o", "{dir}/check.rmeta", "{file}"], code, ".rs") or _check_delimiters(code, language)
    if language == "html":
        try:
            parser = _HTMLCheck()
            parser.feed(code)
            parser.close()
            if parser.failed:
                return {"status": "failed", "method": "html_parser", "message": "Ошибка разбора HTML."}
            return {"status": "limited", "method": "html_parser", "message": "HTML разобран стандартным парсером; соответствие стандарту и корректность DOM полностью не проверялись."}
        except Exception as error:
            return {"status": "failed", "method": "html_parser", "message": "Ошибка HTML: " + str(error)[:180]}
    if language in {"gdscript", "kotlin", "csharp", "dart", "sql", "powershell", "lua", "r", "scala", "perl", "elixir", "haskell", "css", "yaml", "dockerfile", "glsl", "solidity", "other"}:
        if language == "yaml":
            try:
                import yaml  # type: ignore
                yaml.safe_load(code)
                return {"status": "passed", "method": "pyyaml_safe_load", "message": "YAML корректен (без выполнения тегов)."}
            except ImportError:
                pass
            except Exception as error:
                return {"status": "failed", "method": "pyyaml_safe_load", "message": "Ошибка YAML: " + str(error)[:180]}
        return _check_delimiters(code, language)
    return {"status": "limited", "method": "none", "message": "Для выбранного формата доступна только ограниченная проверка."}


def _example_tokens(text: str) -> set[str]:
    stop = {
        "как", "что", "это", "для", "при", "или", "если", "можно", "нужно",
        "сделать", "создать", "написать", "простой", "простая", "простое",
        "в", "на", "из", "по", "с", "и", "the", "how", "to", "a", "an",
    }
    return {
        token for token in re.findall(r"[a-zа-яё0-9_+#.]+", text.lower())
        if len(token) > 1 and token not in stop
    }


def retrieve_coding_examples(
    task: str,
    language: str,
    examples_path: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve curated code examples using lexical coverage and language match."""
    path = examples_path or os.getenv("SWIN_CODING_EXAMPLES_PATH")
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "coding_dialogue.jsonl")
    if not os.path.isfile(path):
        return []

    requested_language = normalize_language(language)
    query_tokens = _example_tokens(task)
    if not query_tokens:
        return []

    ranked: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                if not question or not answer:
                    continue
                example_language = detect_language(question, "auto")
                if requested_language != "other" and example_language != requested_language:
                    continue

                example_tokens = _example_tokens(question)
                intersection = query_tokens.intersection(example_tokens)
                if not intersection:
                    continue
                coverage = len(intersection) / len(query_tokens)
                union_score = len(intersection) / max(1, len(query_tokens.union(example_tokens)))
                # Query coverage gives weight to a narrow, well-matched request;
                # Jaccard prevents a single generic word from dominating.
                score = 0.75 * coverage + 0.25 * union_score
                ranked.append({
                    "question": question,
                    "answer": answer,
                    "score": round(score, 4),
                    "language": example_language,
                    "line": line_number,
                })
    except OSError:
        return []

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:max(1, min(int(limit), 5))]


def _example_code(answer: str) -> str:
    value = answer.strip()
    fence = chr(96) * 3
    match = re.search(
        re.escape(fence) + r"(?:[a-zA-Z0-9_+.#-]+)?\s*\n(.*?)" + re.escape(fence),
        value,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Code samples may contain blank lines inside functions and classes.
    # Remove only a final prose note, not the first blank-line-delimited block.
    note_start = re.search(
        r"\n{2,}(?=(?:Добавь|Назначь|Используй|Проверь|Для сложного|Для production|"
        r"Для публичного|Компилируй|Сохрани|Включи|Перед реальным|Перед запуском|"
        r"Не вставляй|Не храни|Значения передавай)\b)",
        value,
        flags=re.IGNORECASE,
    )
    if note_start:
        value = value[:note_start.start()]
    return value.strip()

def _make_prompt(
    task: str,
    language: str,
    sources: list[dict[str, str]],
    project_context: str,
    block_size: int,
    examples: list[dict[str, Any]] | None = None,
    tokenizer=None,
    response_language: str = "en",
) -> str:
    """Build a prompt using the checkpoint's actual tokenization units."""
    token_limit = max(64, block_size - 4)
    character_tokens = getattr(tokenizer, "vocabulary", None) is not None

    # Use the checkpoint's true token IDs. In particular, BPE is not byte counting.
    def token_length(value: str) -> int:
        encoder = getattr(tokenizer, "encode", None)
        if callable(encoder):
            return len(encoder(value))
        return len(value.encode("utf-8"))

    prefix = f"{response_language_instruction(response_language)}\n"
    task_text = re.sub(r"\s+", " ", task).strip()
    suffix = "\nAssistant:"
    task_label = "User: "
    fixed_length = token_length(prefix + task_label + suffix)
    task_budget = max(1, token_limit - fixed_length)

    if token_length(task_text) > task_budget:
        marker = " ... "
        marker_length = token_length(marker)
        if task_budget <= marker_length:
            task_text = task_text[-task_budget:]
        else:
            usable = task_budget - marker_length
            head_budget = max(1, int(usable * 0.15))
            tail_budget = usable - head_budget
            if character_tokens:
                task_text = task_text[:head_budget] + marker + task_text[-tail_budget:]
            elif tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
                task_data = task_text.encode("utf-8")
                task_text = (
                    task_data[:head_budget].decode("utf-8", errors="ignore")
                    + marker
                    + task_data[-tail_budget:].decode("utf-8", errors="ignore")
                )
            else:
                # BPE token counts are not proportional to bytes. Find the
                # largest head/tail character window that really fits.
                low, high = 1, len(task_text)
                best = marker
                while low <= high:
                    keep = (low + high) // 2
                    head_chars = max(1, int(keep * 0.15))
                    tail_chars = max(0, keep - head_chars)
                    candidate = (
                        task_text[:head_chars] + marker
                        + (task_text[-tail_chars:] if tail_chars else "")
                    )
                    if token_length(candidate) <= task_budget:
                        best = candidate
                        low = keep + 1
                    else:
                        high = keep - 1
                task_text = best

    optional_lines: list[str] = []
    context = re.sub(r"\s+", " ", project_context).strip()
    if context:
        optional_lines.append("Project context: " + context[:300])
    for source in sources[:3]:
        snippet = re.sub(r"\s+", " ", source.get("snippet", "")).strip()
        if snippet:
            optional_lines.append("Reference: " + snippet[:260])
    for example in (examples or [])[:1]:
        snippet = re.sub(r"\s+", " ", _example_code(str(example.get("answer", "")))).strip()
        if snippet:
            optional_lines.append("Verified code example: " + snippet[:150])

    prompt = prefix
    for line in optional_lines:
        candidate = prompt + line + "\n" + task_label + task_text + suffix
        if token_length(candidate) <= token_limit:
            prompt += line + "\n"
    prompt += task_label + task_text + suffix
    if token_length(prompt) > token_limit:
        raise ValueError("Task exceeds SWIN model context window")
    return prompt


def run_code_assist(
    task: str,
    language: str,
    transformer_model,
    research_web: bool = True,
    project_context: str = "",
    show_activity: bool = False,
    response_language: str = "auto",
    text_generator=None,
) -> dict[str, Any]:
    """Research, generate, run independent static checks and report limits."""
    normalized_task = (task or "").strip()
    response_language = resolve_response_language(normalized_task, response_language)
    copy = code_assist_copy(response_language)
    if not normalized_task:
        raise ValueError(copy["empty"])
    if len(normalized_task) > 4000:
        raise ValueError(copy["too_long"])

    language = detect_language(normalized_task, language)

    def localize_result(result: dict[str, Any]) -> dict[str, Any]:
        status = result.get("status", "draft_needs_review")
        validation = result.get("validation") or {}
        validation_status = validation.get("status", "not_run")
        result["response_language"] = response_language
        result["plan"] = [
            copy["plan"][0].format(language=LANGUAGES.get(language, language)),
            copy["plan"][1],
            copy["plan"][2],
        ]
        result["summary"] = copy["summary"].get(status, copy["summary"]["draft_needs_review"])
        if isinstance(validation, dict):
            raw_message = validation.get("message", "")
            if raw_message:
                validation["technical_detail"] = raw_message
            validation["message"] = copy["validation"].get(
                validation_status, copy["validation"]["not_run"]
            )

        warnings = []
        if status == "retrieved_example":
            warnings.extend([copy["warnings"]["retrieved"], copy["warnings"]["adapt"]])
            if validation_status != "passed":
                warnings.append(copy["warnings"]["limited"])
        elif status == "model_not_loaded":
            warnings.append(copy["warnings"]["deploy"])
        elif status == "generation_failed":
            warnings.append(copy["warnings"]["resources"])
        else:
            if validation_status == "limited" or validation_status == "not_supported":
                warnings.append(copy["warnings"]["limited"])
            elif validation_status == "failed":
                warnings.append(copy["warnings"]["failed"])
            if result.get("references"):
                warnings.append(copy["warnings"]["web"])
            warnings.extend([copy["warnings"]["static"], copy["warnings"]["runtime"]])
        result["warnings"] = warnings

        verification = result.get("verification") or {}
        research = verification.get("research")
        if isinstance(research, dict):
            research["message"] = (
                copy["research_found"] if research.get("sources_found", 0)
                else copy["research_none"]
            )
        syntax = verification.get("syntax")
        if isinstance(syntax, dict) and isinstance(validation, dict):
            verification["syntax"] = validation
        runtime = verification.get("runtime_tests")
        if isinstance(runtime, dict):
            runtime["message"] = copy["runtime"]
        cross_check = verification.get("cross_check")
        if isinstance(cross_check, dict):
            cross_check["message"] = (
                copy["cross_found"]
                if cross_check.get("status") in {"references_attached", "local_example"}
                else copy["cross_none"]
            )

        if show_activity:
            for item in result.get("activity", []):
                stage = item.get("stage")
                state = item.get("status")
                if stage == "requirements":
                    item["label"] = copy["activity"]["requirements"]
                elif stage == "retrieval":
                    item["label"] = copy["activity"]["retrieval"].format(count=len(examples))
                elif stage == "research":
                    item["label"] = copy["activity"]["research"].format(count=len(sources))
                elif stage == "planning":
                    item["label"] = copy["activity"]["planning"]
                elif stage == "generation":
                    item["label"] = copy["activity"]["generation"]
                elif stage == "quality_guard":
                    item["label"] = copy["activity"]["quality_guard"]
                elif stage == "repair":
                    item["label"] = copy["activity"]["repair_done" if state == "done" else "repair_failed"]
                elif stage == "validation":
                    item["label"] = copy["validation"].get(validation_status, copy["validation"]["not_run"])
        return result

    activity = [{"stage": "requirements", "status": "done", "label": "Требования определены"}]
    examples = retrieve_coding_examples(normalized_task, language)
    activity.append({
        "stage": "retrieval",
        "status": "done" if examples else "no_match",
        "label": f"Локальная библиотека проверенных примеров: {len(examples)} совпадений",
    })
    sources = search_documentation(normalized_task, language) if research_web else []
    activity.append({
        "stage": "research",
        "status": "done" if sources else ("skipped" if not research_web else "unavailable"),
        "label": f"Поиск по открытому интернету: {len(sources)} источников",
    })
    plan = [
        f"Подготовить решение на {LANGUAGES.get(language, language)}.",
        "Сопоставить найденные справочные материалы и примеры; приоритет — официальной документации.",
        "Сформировать черновик и провести статическую проверку без запуска программы.",
    ]
    research_check = {
        "status": "passed" if sources else ("not_run" if not research_web else "unavailable"),
        "sources_found": len(sources),
        "scope": "broad_public_web",
        "message": "Найдены веб-источники; это не гарантия полноты или достоверности каждой страницы." if sources else "Веб-источники не получены.",
    }

    # For a close match, prefer a curated implementation over free-form
    # generation from the current small checkpoint.
    if examples and examples[0]["score"] >= 0.68:
        code = _example_code(str(examples[0]["answer"]))
        validation = validate_code(code, language)
        activity.append({
            "stage": "retrieval",
            "status": "done",
            "label": "Использован близкий пример из локальной библиотеки",
        })
        warnings = [
            "Это близкий готовый пример, а не полностью адаптированная реализация всего проекта.",
            "Сверь пути узлов, версии библиотек и входные данные со своим проектом.",
            "Синтаксическая проверка не доказывает правильность алгоритма и не заменяет тесты проекта.",
        ]
        if validation["status"] != "passed":
            warnings.append("Автоматическая проверка ограничена или не пройдена; код требует ручной проверки.")
        result = {
            "status": "retrieved_example",
            "language": language,
            "summary": "Найден близкий пример в локальной библиотеке и выполнена доступная проверка.",
            "code": code,
            "validation": validation,
            "verification": {
                "research": research_check,
                "syntax": validation,
                "runtime_tests": {
                    "status": "not_run",
                    "message": "Код не исполнялся на сервере.",
                },
                "cross_check": {
                    "status": "local_example",
                    "message": "Решение основано на локальном примере: " + examples[0]["question"],
                },
            },
            "plan": plan,
            "references": sources,
            "matched_examples": [
                {"question": examples[0]["question"], "score": examples[0]["score"]}
            ],
            "warnings": warnings,
        }
        if show_activity:
            result["activity"] = activity
        return localize_result(result)

    if transformer_model is None and text_generator is None:
        result = {
            "status": "model_not_loaded",
            "language": language,
            "summary": "Поиск завершён, но веса SWIN не загружены на сервер. Код не был сгенерирован.",
            "code": "",
            "validation": {"status": "not_run", "message": "Для генерации требуется SWIN Transformer checkpoint."},
            "verification": {
                "research": research_check,
                "syntax": {"status": "not_run", "message": "Код не создан."},
                "runtime_tests": {"status": "not_run", "message": "Сервер никогда не исполняет пользовательский код."},
            },
            "plan": plan,
            "references": sources,
            "warnings": ["Разверни сервер с models/swin-transformer.pt."],
        }
        if show_activity:
            result["activity"] = activity
        return localize_result(result)

    activity.append({"stage": "planning", "status": "done", "label": "Требования и справочные источники подготовлены"})
    block_size = int(getattr(transformer_model, "block_size", 40000 if text_generator else 256))
    prompt = _make_prompt(
        normalized_task,
        language,
        sources,
        project_context,
        block_size,
        examples,
        tokenizer=getattr(transformer_model, "tokenizer", None),
        response_language=response_language,
    )
    try:
        if text_generator is not None:
            raw = text_generator(prompt, 8000)
        else:
            raw = transformer_model.generate(
                prompt, max_new_tokens=900, temperature=0.12, top_k=12, stop_at_newline=False
            )
    except Exception as error:
        result = {
            "status": "generation_failed",
            "language": language,
            "summary": "Не удалось запустить генерацию кода.",
            "code": "",
            "validation": {"status": "not_run", "message": str(error)[:300]},
            "verification": {"research": research_check, "syntax": {"status": "not_run"}, "runtime_tests": {"status": "not_run"}},
            "plan": plan,
            "references": sources,
            "warnings": ["Проверь загрузку модели и ресурсы сервера."],
        }
        if show_activity:
            result["activity"] = activity
        return localize_result(result)

    activity.append({"stage": "generation", "status": "done", "label": "Черновик кода получен"})
    code = _extract_code(raw)
    if looks_like_garbage_code(code):
        code = ""
        validation = {
            "status": "failed",
            "method": "generation_quality_guard",
            "message": "Генерация содержит признаки бессвязного текста и не будет выдана как код.",
        }
        activity.append({
            "stage": "quality_guard",
            "status": "failed",
            "label": "Бессвязная генерация отклонена до показа пользователю",
        })
    else:
        validation = validate_code(code, language)

    # One conservative repair pass is accepted only if its static result improves.
    if validation["status"] == "failed" and code:
        repair_prompt = (
            f"Исправь только синтаксическую ошибку для языка {LANGUAGES.get(language, language)}. "
            f"Ошибка проверки: {validation['message']}. Верни полный код без пояснений.\nКОД:\n{code}"
        )
        repair_bytes = repair_prompt.encode("utf-8")
        repair_prompt = repair_bytes[-max(64, block_size - 4):].decode("utf-8", "ignore")
        try:
            if text_generator is not None:
                repaired = text_generator(repair_prompt, 8000)
            else:
                repaired = transformer_model.generate(
                    repair_prompt, max_new_tokens=900, temperature=0.0, top_k=1, stop_at_newline=False
                )
            repaired_code = _extract_code(repaired)
            repaired_validation = validate_code(repaired_code, language)
            if repaired_validation["status"] in {"passed", "limited"} and repaired_validation["status"] != validation["status"]:
                code, validation = repaired_code, repaired_validation
                activity.append({"stage": "repair", "status": "done", "label": "Повторная генерация прошла более строгую статическую проверку"})
            else:
                activity.append({"stage": "repair", "status": "failed", "label": "Исправленный вариант не улучшил результат проверки"})
        except Exception:
            activity.append({"stage": "repair", "status": "failed", "label": "Не удалось выполнить повторную генерацию"})

    activity.append({
        "stage": "validation",
        "status": "done" if validation["status"] == "passed" else validation["status"],
        "label": validation["message"],
    })
    warnings: list[str] = []
    if validation["status"] == "limited":
        warnings.append("Проверка ограниченная: полноценный компилятор/парсер для этого языка недоступен.")
    elif validation["status"] == "not_supported":
        warnings.append("Для этого формата доступна только ручная проверка.")
    elif validation["status"] == "failed":
        warnings.append("Код не прошёл статическую проверку. Не используй его как готовое решение.")
    if sources:
        warnings.append("Интернет-источники могут содержать ошибки или устаревшие примеры; сверяй версии библиотек.")
    warnings.append("Синтаксическая проверка не доказывает правильность алгоритма и не заменяет тесты проекта.")
    warnings.append("SWIN проверяет код статически и не запускает его на сервере.")

    verification = {
        "research": research_check,
        "syntax": validation,
        "runtime_tests": {
            "status": "not_run",
            "message": "Программа и тесты не исполнялись: сервер не запускает недоверенный код.",
        },
        "cross_check": {
            "status": "references_attached" if sources else "no_sources",
            "message": "К результату приложены найденные материалы для ручной сверки." if sources else "Нечего сверять: поисковые материалы не получены.",
        },
    }
    status = (
        "generation_rejected" if not code and validation.get("method") == "generation_quality_guard"
        else "empty_generation" if not code
        else "draft_validated" if validation["status"] == "passed"
        else "draft_needs_review"
    )
    result = {
        "status": status,
        "language": language,
        "summary": (
            "Модель выдала бессвязный текст вместо кода. Результат отклонён; уточни задачу или используй близкий проверенный пример."
            if status == "generation_rejected"
            else "Черновик создан и прошёл доступную статическую проверку."
            if validation["status"] == "passed"
            else "Черновик создан; результат проверки ограничен или требует исправления."
        ),
        "code": code,
        "validation": validation,
        "verification": verification,
        "plan": plan,
        "references": sources,
        "matched_examples": [
            {"question": item["question"], "score": item["score"]}
            for item in examples
        ],
        "warnings": warnings,
    }
    if show_activity:
        result["activity"] = activity
    return localize_result(result)
