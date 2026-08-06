"""Text normalization and NLP processing."""

from __future__ import annotations

import importlib.metadata
import re
import unicodedata
from typing import Protocol

from agentic_rag.errors import BuildError
from agentic_rag.substrate.models import (
    AbbreviationLink,
    EntityMentionDraft,
    ProcessedDocument,
    ProcessedSentence,
)

_WHITESPACE = re.compile(r"\s+")
_SURROUNDING_PUNCTUATION = re.compile(r"^\W+|\W+$", flags=re.UNICODE)


def normalize_document_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_entity_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    normalized = _SURROUNDING_PUNCTUATION.sub("", normalized)
    return normalized.strip()


class DocumentProcessor(Protocol):
    segmenter_name: str
    segmenter_version: str | None
    ner_name: str
    ner_version: str | None
    abbreviation_name: str | None
    abbreviation_version: str | None
    chunk_tokenizer_name: str

    def process(self, text: str) -> ProcessedDocument:
        ...


class SpacyDocumentProcessor:
    """One-pass sentence segmentation, NER, and optional abbreviations."""

    def __init__(self, model_name: str, enable_abbreviations: bool = True) -> None:
        try:
            import spacy
        except ImportError as exc:
            raise BuildError("spaCy is required for the default NLP processor") from exc
        try:
            self._nlp = spacy.load(model_name)
        except OSError as exc:
            raise BuildError(
                f"spaCy model {model_name!r} is not installed. Run "
                f"`uv run python -m spacy download {model_name}`."
            ) from exc

        self.segmenter_name = f"spacy:{model_name}"
        self.ner_name = f"spacy:{model_name}"
        try:
            model_version = importlib.metadata.version(model_name)
        except importlib.metadata.PackageNotFoundError:
            model_version = self._nlp.meta.get("version")
        self.segmenter_version = model_version
        self.ner_version = model_version
        self.chunk_tokenizer_name = f"spacy:{model_name}"
        self.abbreviation_name = None
        self.abbreviation_version = None

        if enable_abbreviations:
            try:
                import scispacy  # noqa: F401
                from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
            except ImportError as exc:
                raise BuildError(
                    "scispaCy is required when abbreviation detection is enabled"
                ) from exc
            if "abbreviation_detector" not in self._nlp.pipe_names:
                self._nlp.add_pipe("abbreviation_detector", last=True)
            self.abbreviation_name = "scispacy:AbbreviationDetector"
            try:
                self.abbreviation_version = importlib.metadata.version("scispacy")
            except importlib.metadata.PackageNotFoundError:
                self.abbreviation_version = None

    def process(self, text: str) -> ProcessedDocument:
        doc = self._nlp(text)
        sentence_spans = list(doc.sents)
        sentence_index_by_token: dict[int, int] = {}
        sentence_texts: list[str] = []
        sentence_token_counts: list[int] = []
        sentence_start_chars: list[int] = []
        sentence_mentions: list[list[EntityMentionDraft]] = []
        ner_type_by_span: dict[tuple[int, int], str | None] = {}

        for sentence_index, sent in enumerate(sentence_spans):
            raw = sent.text
            leading = len(raw) - len(raw.lstrip())
            trailing_text = raw.strip()
            sentence_start_char = sent.start_char + leading
            sentence_end_char = sentence_start_char + len(trailing_text)
            mentions: list[EntityMentionDraft] = []
            for entity in sent.ents:
                if entity.start_char < sentence_start_char:
                    continue
                if entity.end_char > sentence_end_char:
                    continue
                start = entity.start_char - sentence_start_char
                end = entity.end_char - sentence_start_char
                surface = trailing_text[start:end]
                mentions.append(
                    EntityMentionDraft(
                        sentence_index=sentence_index,
                        surface_form=surface,
                        start=start,
                        end=end,
                        entity_type=entity.label_ or None,
                    )
                )
                ner_type_by_span[(entity.start_char, entity.end_char)] = (
                    entity.label_ or None
                )
            for token_index in range(sent.start, sent.end):
                sentence_index_by_token[token_index] = sentence_index
            sentence_texts.append(trailing_text)
            sentence_token_counts.append(max(1, len(sent)))
            sentence_start_chars.append(sentence_start_char)
            sentence_mentions.append(mentions)

        abbreviations: list[AbbreviationLink] = []
        if self.abbreviation_name is not None:
            for abbreviation in doc._.abbreviations:
                long_form = abbreviation._.long_form
                source_index = sentence_index_by_token.get(
                    long_form.start,
                    sentence_index_by_token.get(abbreviation.start, 0),
                )
                abbreviations.append(
                    AbbreviationLink(
                        short_form=abbreviation.text,
                        long_form=long_form.text,
                        source_sentence_index=source_index,
                    )
                )
                abbreviation_type = ner_type_by_span.get(
                    (long_form.start_char, long_form.end_char),
                    ner_type_by_span.get(
                        (abbreviation.start_char, abbreviation.end_char)
                    ),
                )
                for span in (long_form, abbreviation):
                    span_sentence_index = sentence_index_by_token.get(span.start)
                    if span_sentence_index is None:
                        continue
                    local_start = (
                        span.start_char
                        - sentence_start_chars[span_sentence_index]
                    )
                    local_end = (
                        span.end_char - sentence_start_chars[span_sentence_index]
                    )
                    sentence_text = sentence_texts[span_sentence_index]
                    if (
                        local_start < 0
                        or local_end > len(sentence_text)
                        or sentence_text[local_start:local_end] != span.text
                    ):
                        continue
                    existing_spans = {
                        (item.start, item.end)
                        for item in sentence_mentions[span_sentence_index]
                    }
                    if (local_start, local_end) in existing_spans:
                        continue
                    sentence_mentions[span_sentence_index].append(
                        EntityMentionDraft(
                            sentence_index=span_sentence_index,
                            surface_form=span.text,
                            start=local_start,
                            end=local_end,
                            entity_type=abbreviation_type,
                        )
                    )
        unique_abbreviations = {
            (item.short_form, item.long_form, item.source_sentence_index): item
            for item in abbreviations
        }
        processed_sentences = [
            ProcessedSentence(
                text=sentence_text,
                token_count=sentence_token_counts[sentence_index],
                mentions=tuple(
                    sorted(
                        sentence_mentions[sentence_index],
                        key=lambda item: (
                            item.start,
                            item.end,
                            item.surface_form,
                        ),
                    )
                ),
            )
            for sentence_index, sentence_text in enumerate(sentence_texts)
        ]
        return ProcessedDocument(
            sentences=tuple(processed_sentences),
            abbreviations=tuple(
                unique_abbreviations[key] for key in sorted(unique_abbreviations)
            ),
        )
