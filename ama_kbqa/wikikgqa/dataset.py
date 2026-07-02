"""Loader for the WikiKGQA extended QALD-JSON datasets.

Handles both the labelled training file (gold ``query.sparql`` + populated
``answers``) and the test files (placeholder strings in ``query``/``answers``
that the system must fill). A single :class:`WikiKGQAQuestion` normalises both.

Format reference (one entry)::

    {
      "id": 0,
      "question": [{"string": "...", "language": "en"}, {"string": "...", "language": "es"}],
      "query": {"sparql": "SELECT ... WHERE { ... }"},
      "mentions": [
        {"string": "Eleven", "language": "en", "entity": "http://www.wikidata.org/entity/Q27955792"},
        {"string": "played", "language": "en",
         "entity": "...", "property": "http://www.wikidata.org/prop/P161", "inverse": true}
      ],
      "answers": [{"head": {"vars": ["obj2"]}, "results": {"bindings": [...]}}]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Placeholder text the organisers ship in the unlabelled test files; used to tell
# a real gold value apart from "fill this in yourself".
_PLACEHOLDER_MARKERS = ("Optional", "Mandatory", "fill here", "preferably add")

ENTITY_PREFIX = "http://www.wikidata.org/entity/"
# Properties appear under several Wikidata namespaces; strip any of them to the bare Pxx.
_PROPERTY_PREFIXES = (
    "http://www.wikidata.org/prop/direct/",
    "http://www.wikidata.org/prop/statement/",
    "http://www.wikidata.org/prop/qualifier/",
    "http://www.wikidata.org/prop/",
)


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and any(m in value for m in _PLACEHOLDER_MARKERS)


def strip_entity(uri: str | None) -> str | None:
    """Return the bare ``Qxxx`` id for an entity URI (pass-through if already bare)."""
    if not uri:
        return None
    if uri.startswith(ENTITY_PREFIX):
        return uri[len(ENTITY_PREFIX):]
    return uri


def strip_property(uri: str | None) -> str | None:
    """Return the bare ``Pxxx`` id for a property URI (pass-through if already bare)."""
    if not uri:
        return None
    for prefix in _PROPERTY_PREFIXES:
        if uri.startswith(prefix):
            return uri[len(prefix):]
    return uri


@dataclass
class Mention:
    """A single entity/property mention annotation for one language."""

    string: str
    language: str
    entity: str | None = None        # bare Qxxx
    entity_uri: str | None = None     # full URI as given
    property: str | None = None       # bare Pxxx
    property_uri: str | None = None    # full URI as given
    inverse: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Mention":
        return cls(
            string=raw.get("string", ""),
            language=raw.get("language", ""),
            entity=strip_entity(raw.get("entity")),
            entity_uri=raw.get("entity"),
            property=strip_property(raw.get("property")),
            property_uri=raw.get("property"),
            inverse=bool(raw.get("inverse", False)),
        )


@dataclass
class WikiKGQAQuestion:
    """A normalised WikiKGQA question, gold-bearing or not."""

    id: int
    questions: dict[str, str]                 # language -> question string
    mentions: list[Mention] = field(default_factory=list)
    gold_sparql: str | None = None            # None for test entries
    gold_answer: dict[str, Any] | None = None  # raw SPARQL-JSON (QALD format), None otherwise
    gold_values: list[str] | None = None       # bare answers (no-mentions splits), None otherwise
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gold(self) -> bool:
        return self.gold_answer is not None or self.gold_values is not None

    @property
    def has_mentions(self) -> bool:
        return len(self.mentions) > 0

    def question(self, language: str = "en") -> str:
        """Question string for a language, falling back to any available one."""
        if language in self.questions:
            return self.questions[language]
        return next(iter(self.questions.values()), "")

    def mentions_for(self, language: str = "en") -> list[Mention]:
        return [m for m in self.mentions if m.language == language]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "WikiKGQAQuestion":
        questions = {
            q.get("language", ""): q.get("string", "")
            for q in raw.get("question", [])
            if q.get("string")
        }

        # gold query lives at query.sparql (QALD format) or top-level sparql (no-mentions).
        sparql = (raw.get("query") or {}).get("sparql") or raw.get("sparql")
        if _is_placeholder(sparql):
            sparql = None

        gold_answer = None
        gold_values = None
        answers = raw.get("answers") or []
        if answers and not _is_placeholder(answers[0]):
            if isinstance(answers[0], dict):
                # QALD format: the SPARQL-JSON object is stored directly in answers[0].
                gold_answer = answers[0]
            else:
                # No-mentions splits: answers is a bare list of values ("Q..."/"3"/...).
                gold_values = [str(a) for a in answers]

        mentions = [Mention.from_raw(m) for m in raw.get("mentions", [])]

        return cls(
            id=raw.get("id", -1),
            questions=questions,
            mentions=mentions,
            gold_sparql=sparql,
            gold_answer=gold_answer,
            gold_values=gold_values,
            raw=raw,
        )


@dataclass
class WikiKGQADataset:
    """A loaded WikiKGQA dataset file."""

    dataset_id: str
    questions: list[WikiKGQAQuestion]

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    @property
    def has_gold(self) -> bool:
        return any(q.has_gold for q in self.questions)

    @classmethod
    def load(cls, path: str | Path) -> "WikiKGQADataset":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        meta = data.get("dataset", {})
        questions = [WikiKGQAQuestion.from_raw(q) for q in data.get("questions", [])]
        return cls(dataset_id=meta.get("id", str(path)), questions=questions)
