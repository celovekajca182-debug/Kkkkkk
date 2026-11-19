import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote


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
WIKI_DOCUMENT_SEPARATOR = "<|document|>"


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
    """Curated local answers plus retrieval over the prepared FineWiki/Wikipedia corpus."""

    def __init__(
        self,
        path: str | Path,
        wiki_path: str | Path | None = None,
        max_wiki_chunks: int = 12000,
    ):
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

        # Reviewed dialogue examples remain exact-match retrieval candidates.
        dialogue_path = self.path.with_name("dialogue.jsonl")
        if dialogue_path.is_file():
            known_questions = {
                str(item.get("question", "")).strip().casefold()
                for item in self.entries
            }
            for line in dialogue_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = str(pair.get("question", "")).strip()
                answer = str(pair.get("answer", "")).strip()
                if not question or not answer or question.casefold() in known_questions:
                    continue
                self.entries.append({
                    "title": "Проверенный диалог: " + question[:100],
                    "question": question,
                    "keywords": tokenize(question),
                    "answer": answer,
                })
                known_questions.add(question.casefold())

        self.index, self.document_frequency = self._build_index(
            self.entries,
            lambda item: " ".join([
                item.get("title", ""),
                item.get("question", ""),
                " ".join(item.get("keywords", [])),
            ]),
        )

        if wiki_path is None:
            wiki_path = self.path.with_name("finewiki_ru_train.txt")
        self.wiki_path = Path(wiki_path)
        self.wiki_entries = self._load_wikipedia_chunks(
            self.wiki_path, max_chunks=max_wiki_chunks
        )
        self.wiki_index, self.wiki_document_frequency = self._build_index(
            self.wiki_entries,
            lambda item: item["title"] + " " + item["snippet"],
        )
        self.wiki_articles = len({item["article"] for item in self.wiki_entries})
        self.size = len(self.entries) + len(self.wiki_entries)

    @staticmethod
    def _build_index(entries, text_for_item):
        index = []
        frequency = Counter()
        for item in entries:
            counts = Counter(tokenize(text_for_item(item)))
            index.append(counts)
            frequency.update(counts.keys())
        return index, frequency

    @staticmethod
    def _load_wikipedia_chunks(path: Path, max_chunks: int) -> list[dict]:
        if not path.is_file() or max_chunks <= 0:
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return []

        documents = raw.split(WIKI_DOCUMENT_SEPARATOR)
        chunks: list[dict] = []
        seen_titles: set[str] = set()
        max_chunks = max(1, min(int(max_chunks), 30000))

        for document in documents:
            document = document.strip()
            if not document:
                continue
            lines = document.splitlines()
            title = lines[0].strip()[:180] if lines else ""
            body = "\n".join(lines[1:]).strip()
            if not title or len(body) < 100:
                continue
            title_key = title.casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            # Keep paragraph boundaries and cap each passage for the small SWIN
            # context window. Long paragraphs are split into bounded sentences.
            paragraphs = [
                re.sub(r"\\s+", " ", part).strip()
                for part in re.split(r"\\n\\s*\\n", body)
                if len(part.strip()) >= 50
            ]
            passage = ""
            for paragraph in paragraphs:
                while len(paragraph) > 1100:
                    split_at = paragraph.rfind(". ", 0, 1050)
                    if split_at < 400:
                        split_at = paragraph.rfind(" ", 0, 1050)
                    if split_at < 1:
                        split_at = 1050
                    candidate = paragraph[:split_at + 1].strip()
                    if len(candidate) >= 80:
                        chunks.append({
                            "title": title,
                            "article": title,
                            "snippet": candidate[:1100],
                        })
                    paragraph = paragraph[split_at + 1:].strip()
                    if len(chunks) >= max_chunks:
                        return chunks
                if passage and len(passage) + len(paragraph) > 1100:
                    chunks.append({
                        "title": title,
                        "article": title,
                        "snippet": passage[:1100],
                    })
                    passage = ""
                    if len(chunks) >= max_chunks:
                        return chunks
                passage = (passage + " " + paragraph).strip()
            if len(passage) >= 80:
                chunks.append({
                    "title": title,
                    "article": title,
                    "snippet": passage[:1100],
                })
            if len(chunks) >= max_chunks:
                break
        return chunks

    @staticmethod
    def _rank(query: str, entries, index, document_frequency, limit: int, min_score: float):
        query_terms = Counter(tokenize(query))
        if not query_terms or not entries:
            return []
        total_docs = max(1, len(entries))
        ranked = []
        for item, document_tokens in zip(entries, index):
            matched = set(query_terms).intersection(document_tokens)
            if not matched:
                continue
            weighted = 0.0
            covered = 0.0
            for token in matched:
                idf = math.log(1.0 + (total_docs + 1.0) /
                               (document_frequency.get(token, 0) + 0.5))
                tf = document_tokens[token]
                weighted += idf * (1.0 + math.log(tf))
                covered += idf
            total_weight = sum(
                math.log(1.0 + (total_docs + 1.0) /
                         (document_frequency.get(token, 0) + 0.5))
                for token in query_terms
            ) or 1.0
            score = (covered / total_weight) * min(1.0, weighted / max(covered, 0.001))
            if score >= min_score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return ranked[:max(1, min(int(limit), 10))]

    def search(self, query: str, min_score: float = 0.12):
        ranked = self._rank(
            query, self.entries, self.index, self.document_frequency, 1, min_score
        )
        if not ranked:
            return None
        score, item = ranked[0]
        return {
            "title": item["title"],
            "answer": item["answer"],
            "score": score,
        }

    def search_many(self, query: str, limit: int = 4, min_score: float = 0.10) -> list[dict]:
        """Return attributed Wikipedia passages for retrieval-augmented generation."""
        ranked = self._rank(
            query,
            self.wiki_entries,
            self.wiki_index,
            self.wiki_document_frequency,
            limit,
            min_score,
        )
        references = []
        seen_articles = set()
        for score, item in ranked:
            article = item["article"]
            if article.casefold() in seen_articles:
                continue
            seen_articles.add(article.casefold())
            references.append({
                "title": article,
                "snippet": item["snippet"],
                "url": "https://ru.wikipedia.org/wiki/" + quote(article.replace(" ", "_")),
                "source_type": "FineWiki / Википедия (CC BY-SA 4.0)",
                "trust": "encyclopedia",
                "score": round(score, 3),
            })
        return references
