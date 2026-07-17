"""src/graph/neo4j_client.py — Neo4j driver with batch upsert + traversal helpers"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from neo4j import GraphDatabase, Session
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)


class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        logger.info("Neo4j connected: %s", uri)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self._driver.session(database=self._database) as s:
            yield s

    def close(self) -> None:
        self._driver.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Create constraints and indexes on first run."""
        stmts = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
        ]
        with self.session() as s:
            for stmt in stmts:
                s.run(stmt)
        logger.info("Neo4j schema ready")

    # ── Write ─────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        if not nodes:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $nodes AS n
                MERGE (e:Entity {id: n.id})
                SET e += n.props, e.label = n.label
                """,
                nodes=[{"id": n.id, "label": n.label, "props": n.properties} for n in nodes],
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def upsert_relations(self, relations: list[GraphRelation]) -> None:
        if not relations:
            return
        with self.session() as s:
            s.run(
                """
                UNWIND $rels AS r
                MATCH (a:Entity {id: r.src}), (b:Entity {id: r.tgt})
                MERGE (a)-[rel:RELATES {type: r.rtype}]->(b)
                SET rel += r.props
                """,
                rels=[
                    {"src": r.source_id, "tgt": r.target_id,
                     "rtype": r.relation_type, "props": r.properties}
                    for r in relations
                ],
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def neighbourhood(self, entity_id: str, hops: int = 2, limit: int = 40) -> list[dict]:
        with self.session() as s:
            result = s.run(
                f"""
                MATCH path = (start:Entity {{id: $eid}})-[*1..{hops}]-(nb)
                UNWIND relationships(path) AS rel
                RETURN startNode(rel).name AS src, type(rel) AS rtype, endNode(rel).name AS tgt
                LIMIT {limit}
                """,
                eid=entity_id,
            )
            return [dict(r) for r in result]

    def fulltext_search(self, query: str, limit: int = 5) -> list[dict]:
        with self.session() as s:
            result = s.run(
                """
                CALL db.index.fulltext.queryNodes('entity_name', $q)
                YIELD node, score
                RETURN node.id AS id, node.name AS name, node.label AS label, score
                ORDER BY score DESC LIMIT $limit
                """,
                q=query, limit=limit,
            )
            return [dict(r) for r in result]

    def run(self, cypher: str, **params: Any) -> list[dict]:
        with self.session() as s:
            return [dict(r) for r in s.run(cypher, **params)]
