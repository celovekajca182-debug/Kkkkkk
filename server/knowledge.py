import json
import math
import re
from pathlib import Path
from collections import Counter


STOP_WORDS = {
    "и", "в", "во", "на", "с", "со", "для", "по", "из", "к", "у", "от",
    "это", "как", "что", "где", "когда", "ли", "а", "но", "или", "не",
    "я", "ты", "мы", "вы", "он", "она", "они", "мне", "моя", "мой",
    "the", "a", "an", "is", "are", "to", "of", "for", "and", "or", "how",
}
SUFFIXES = (
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими",
    "ать", "ять", "ить", "еть", "ешь", "ете", "ают", "яют", "ами",
    "ах", "ях", "ов", "ев", "ей", "ый", "ий", "ая", "яя", "ое", "ее",
    "ые", "ие", "ам", "ям", "ом", "ем", "ы", "и", "а", "я", "у", "ю", "е",
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zа-яё0-9_+#.]+", text.lower())
    result = []
    for word in words:
        if len(word) > 4:
            for suffix in SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    word = word[:-len(suffix)]
                    break
        if len(word) > 1 and word not in STOP_WORDS:
            result.append(word)
    return result


class KnowledgeBase:
    """Небольшой локальный поиск по проверенным материалам, без внешних AI API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            raw = []
        if not isinstance(raw, list):
            raise ValueError("Файл базы знаний должен содержать JSON-массив.")
        self.entries = [
            item for item in raw
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("answer"), str)
        ]
        self.index = []
        self.document_frequency = Counter()
        for item in self.entries:
            searchable = " ".join([
                item.get("title", ""),
                item.get("question", ""),
                " ".join(item.get("keywords", [])),
            ])
            tokens = tokenize(searchable)
            counts = Counter(tokens)
            self.index.append(counts)
            self.document_frequency.update(counts.keys())
        self.size = len(self.entries)

    def search(self, query: str, min_score: float = 0.12):
        query_tokens = set(tokenize(query))
        if not query_tokens or not self.entries:
            return None

        total_docs = len(self.entries)
        weights = {
            token: math.log(1.0 + (total_docs + 1.0) /
                            (self.document_frequency.get(token, 0) + 0.5))
            for token in query_tokens
        }
        total_weight = sum(weights.values()) or 1.0

        best_item = None
        best_score = 0.0
        for item, document_tokens in zip(self.entries, self.index):
            matched = query_tokens.intersection(document_tokens)
            score = sum(weights[token] for token in matched) / total_weight
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None or best_score < min_score:
            return None
        return {
            "title": best_item["title"],
            "answer": best_item["answer"],
            "score": best_score,
        }
