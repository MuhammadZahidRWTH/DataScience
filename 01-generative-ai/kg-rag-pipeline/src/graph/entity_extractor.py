"""src/graph/entity_extractor.py — NER + relation extraction → GraphNode/GraphRelation"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import spacy

from src.graph.neo4j_client import GraphNode, GraphRelation

# spaCy label → human label
LABEL_MAP = {
    "ORG": "organisation",
    "PERSON": "person",
    "GPE": "location",
    "LOC": "location",
    "PRODUCT": "product",
    "DATE": "date",
    "MONEY": "financial",
    "LAW": "regulation",
    "WORK_OF_ART": "work",
    "EVENT": "event",
}

RELATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(acquired|bought|purchased)\b", re.I), "ACQUIRED"),
    (re.compile(r"\b(partnered with|collaborated with)\b", re.I), "PARTNERED_WITH"),
    (re.compile(r"\b(founded|established|created)\b", re.I), "FOUNDED"),
    (re.compile(r"\b(ceo of|leads|headed by|manages)\b", re.I), "LEADS"),
    (re.compile(r"\b(uses|utilises|powered by|built on)\b", re.I), "USES"),
    (re.compile(r"\b(competes with|rival of)\b", re.I), "COMPETES_WITH"),
    (re.compile(r"\b(subsidiary of|owned by|part of)\b", re.I), "SUBSIDIARY_OF"),
    (re.compile(r"\b(reported|announced|said)\b", re.I), "REPORTED"),
]


@dataclass
class ExtractionResult:
    nodes: list[GraphNode]
    relations: list[GraphRelation]
    raw_entities: list[dict[str, Any]] = field(default_factory=list)


def _uid(text: str, label: str, doc_id: str) -> str:
    return hashlib.sha1(f"{doc_id}::{label}::{text.lower()}".encode()).hexdigest()[:16]


def _classify_relation(sentence: str) -> str:
    for pattern, rtype in RELATION_PATTERNS:
        if pattern.search(sentence):
            return rtype
    return "CO_OCCURS_WITH"


class EntityExtractor:
    """
    Extracts named entities and co-occurrence relations using spaCy en_core_web_sm.

    For production, swap spaCy NER for GLiNER (zero-shot) by changing _extract_entities.
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def extract(self, text: str, doc_id: str) -> ExtractionResult:
        doc = self.nlp(text)

        # ── Entities ──────────────────────────────────────────────────────────
        seen_ids: set[str] = set()
        raw: list[dict] = []
        nodes: list[GraphNode] = []

        for ent in doc.ents:
            label = LABEL_MAP.get(ent.label_, ent.label_.lower())
            uid = _uid(ent.text, label, doc_id)
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
            raw.append({"id": uid, "text": ent.text, "label": label})
            nodes.append(GraphNode(
                id=uid,
                label=label,
                properties={"name": ent.text, "source_doc": doc_id},
            ))

        # ── Relations (sentence-level co-occurrence) ──────────────────────────
        relations: list[GraphRelation] = []
        seen_rels: set[tuple] = set()

        for sent in doc.sents:
            sent_ents = [e for e in raw if e["text"].lower() in sent.text.lower()]
            rtype = _classify_relation(sent.text)
            for i, src in enumerate(sent_ents):
                for tgt in sent_ents[i + 1:]:
                    key = (src["id"], tgt["id"], rtype)
                    if key not in seen_rels:
                        seen_rels.add(key)
                        relations.append(GraphRelation(
                            source_id=src["id"],
                            target_id=tgt["id"],
                            relation_type=rtype,
                            properties={"sentence": sent.text[:200], "source_doc": doc_id},
                        ))

        return ExtractionResult(nodes=nodes, relations=relations, raw_entities=raw)
