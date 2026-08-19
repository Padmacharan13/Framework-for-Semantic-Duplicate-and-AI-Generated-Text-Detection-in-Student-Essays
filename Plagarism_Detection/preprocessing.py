"""
preprocessing.py
Cleaning, sentence/paragraph segmentation, and lemmatization.

Optimised for throughput: batches every spaCy call via nlp.pipe() instead
of looping row-by-row, and disables pipeline components that aren't needed
for sentence splitting + lemmatization.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from config import SPACY_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_RE_WHITESPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NON_PRINTABLE = re.compile(r"[^\x20-\x7E\n]")
_RE_MULTI_PUNCT = re.compile(r"([!?.,])\1{1,}")


@dataclass
class ProcessedText:
    original: str
    cleaned: str
    paragraphs: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    lemmatized_sentences: list[str] = field(default_factory=list)


class TextPreprocessor:
    def __init__(self, model_name: str = SPACY_MODEL, n_process: int = 1) -> None:
        self.n_process = n_process
        self.nlp: Language = self._load_spacy(model_name)

    @staticmethod
    def _load_spacy(model_name: str) -> Language:
        try:
            nlp = spacy.load(model_name, disable=["ner", "parser"])
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                model_name, model_name,
            )
            raise
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        if not nlp.has_pipe("lemmatizer"):
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            nlp.initialize()
        return nlp

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = _RE_URL.sub(" ", text)
        text = _RE_NON_PRINTABLE.sub(" ", text)
        text = _RE_MULTI_PUNCT.sub(r"\1", text)
        text = text.lower().strip()
        text = _RE_WHITESPACE.sub(" ", text)
        return text

    @staticmethod
    def segment_paragraphs(raw_text: str) -> list[str]:
        if not isinstance(raw_text, str):
            return []
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        return paras if paras else ([raw_text.strip()] if raw_text.strip() else [])

    def segment_sentences_batch(self, texts: Iterable[str]) -> list[list[str]]:
        docs: Iterator[Doc] = self.nlp.pipe(texts, batch_size=256, n_process=self.n_process)
        return [[s.text.strip() for s in doc.sents if s.text.strip()] for doc in docs]

    def lemmatize_batch(self, texts: Iterable[str]) -> list[str]:
        docs: Iterator[Doc] = self.nlp.pipe(texts, batch_size=256, n_process=self.n_process)
        out = []
        for doc in docs:
            lemmas = [t.lemma_ for t in doc if not t.is_space and not t.is_punct]
            out.append(" ".join(lemmas))
        return out

    def process_column(self, texts: Iterable[str]) -> list[ProcessedText]:
        """Efficiently process an entire column/list of raw texts in batched spaCy passes."""
        texts = list(texts)
        cleaned_texts = [self.clean_text(t) for t in texts]
        sentence_lists = self.segment_sentences_batch(cleaned_texts)

        flat_sentences: list[str] = []
        doc_boundaries: list[tuple[int, int]] = []
        cursor = 0
        for sents in sentence_lists:
            start = cursor
            flat_sentences.extend(sents)
            cursor += len(sents)
            doc_boundaries.append((start, cursor))

        flat_lemmas = self.lemmatize_batch(flat_sentences) if flat_sentences else []

        results: list[ProcessedText] = []
        for raw, cleaned, sents, (start, end) in zip(texts, cleaned_texts, sentence_lists, doc_boundaries):
            results.append(
                ProcessedText(
                    original=raw,
                    cleaned=cleaned,
                    paragraphs=self.segment_paragraphs(raw),
                    sentences=sents,
                    lemmatized_sentences=flat_lemmas[start:end],
                )
            )
        return results


if __name__ == "__main__":
    sample_texts = [
        "The automobile is rapid!!! It moves down the road.",
        "The vehicle moves fast. It travels along the street.",
    ]
    pre = TextPreprocessor()
    for p in pre.process_column(sample_texts):
        print("CLEANED:", p.cleaned)
        print("SENTENCES:", p.sentences)
        print("---")