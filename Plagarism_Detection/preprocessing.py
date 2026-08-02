"""
preprocessing.py
Cleaning, sentence/paragraph segmentation, and lemmatization.

Optimised for throughput:
  - uses nlp.pipe(..., batch_size=..., n_process=...) instead of calling
    nlp() in a Python loop (10-50x faster on multi-thousand row datasets)
  - disables unused spaCy pipeline components (ner, parser not needed for
    sentence splitting + lemmatization -> big speedup)
  - regex cleaning compiled once at module load, not per call
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

# ---- Precompiled regex patterns (compiled once, reused everywhere) --------
_RE_WHITESPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_NON_PRINTABLE = re.compile(r"[^\x20-\x7E\n]")
_RE_MULTI_PUNCT = re.compile(r"([!?.,])\1{1,}")


@dataclass
class ProcessedText:
    """Container for a single processed document."""
    original: str
    cleaned: str
    paragraphs: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    lemmatized_sentences: list[str] = field(default_factory=list)


class TextPreprocessor:
    """
    Batch-oriented preprocessor. Instantiate ONCE and reuse across the whole
    dataset (loading the spaCy model is expensive - do it a single time).
    """

    def __init__(self, model_name: str = SPACY_MODEL, n_process: int = 1) -> None:
        self.n_process = n_process
        self.nlp: Language = self._load_spacy(model_name)

    @staticmethod
    def _load_spacy(model_name: str) -> Language:
        """Load spaCy with only the components we actually need."""
        try:
            nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                model_name, model_name,
            )
            raise
        # Re-enable a lightweight lemmatizer (lookup mode is much faster than
        # the full rule-based one and is sufficient for this task).
        if "lemmatizer" not in nlp.pipe_names:
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            nlp.initialize()
        return nlp

    # ---- Cleaning -----------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Lowercasing, URL stripping, non-printable removal, whitespace collapse."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = _RE_URL.sub(" ", text)
        text = _RE_NON_PRINTABLE.sub(" ", text)
        text = _RE_MULTI_PUNCT.sub(r"\1", text)
        text = text.lower().strip()
        text = _RE_WHITESPACE.sub(" ", text)
        return text

    # ---- Segmentation ---------------------------------------------------

    @staticmethod
    def segment_paragraphs(raw_text: str) -> list[str]:
        """Split on blank lines. Falls back to the whole text as one paragraph."""
        if not isinstance(raw_text, str):
            return []
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        return paras if paras else ([raw_text.strip()] if raw_text.strip() else [])

    def segment_sentences_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """
        Batch sentence segmentation. Pass ALL texts you need to split at once
        rather than calling this per-row - nlp.pipe amortises tokenizer/model
        overhead across the whole batch.
        """
        docs: Iterator[Doc] = self.nlp.pipe(
            texts, batch_size=256, n_process=self.n_process
        )
        return [[sent.text.strip() for sent in doc.sents if sent.text.strip()] for doc in docs]

    def lemmatize_batch(self, texts: Iterable[str]) -> list[str]:
        """Batch lemmatization -> space-joined lemma string per input text."""
        docs: Iterator[Doc] = self.nlp.pipe(
            texts, batch_size=256, n_process=self.n_process
        )
        out = []
        for doc in docs:
            lemmas = [t.lemma_ for t in doc if not t.is_space and not t.is_punct]
            out.append(" ".join(lemmas))
        return out

    # ---- Full pipeline for a single text --------------------------------

    def process_one(self, raw_text: str) -> ProcessedText:
        cleaned = self.clean_text(raw_text)
        paragraphs = self.segment_paragraphs(raw_text)
        sentences = self.segment_sentences_batch([cleaned])[0]
        lemmatized = self.lemmatize_batch(sentences) if sentences else []
        return ProcessedText(
            original=raw_text,
            cleaned=cleaned,
            paragraphs=paragraphs,
            sentences=sentences,
            lemmatized_sentences=lemmatized,
        )

    # ---- Full pipeline for a whole column of texts (USE THIS ONE) -------

    def process_column(self, texts: Iterable[str]) -> list[ProcessedText]:
        """
        Efficiently process an entire column/list of raw texts.
        This is the method the data-loading scripts call - it batches every
        spaCy call instead of looping process_one() row by row.
        """
        texts = list(texts)
        cleaned_texts = [self.clean_text(t) for t in texts]

        # one batched spaCy pass for sentence splitting across ALL docs
        sentence_lists = self.segment_sentences_batch(cleaned_texts)

        # flatten all sentences from all docs into one big batch for lemmatization,
        # then re-split back per document - avoids N separate small nlp.pipe calls
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
        for raw, cleaned, sents, (start, end) in zip(
            texts, cleaned_texts, sentence_lists, doc_boundaries
        ):
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
    # Quick smoke test
    sample_texts = [
        "The automobile is rapid!!! It moves down the road.\n\nA second paragraph here.",
        "The vehicle moves fast. It travels along the street.",
    ]
    pre = TextPreprocessor()
    processed = pre.process_column(sample_texts)
    for p in processed:
        print("CLEANED:", p.cleaned)
        print("SENTENCES:", p.sentences)
        print("LEMMAS:", p.lemmatized_sentences)
        print("---")