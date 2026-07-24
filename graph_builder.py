"""Build directed, labelled idea graphs from extracted relationships."""

from __future__ import annotations

import networkx as nx

from extractor import Relationship, extract_relationships


def build_idea_graph(text: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    for relationship in extract_relationships(text):
        graph.add_edge(
            relationship.subject,
            relationship.object,
            relation=relationship.relation,
        )
    return graph


def graph_relationships(graph: nx.DiGraph) -> set[Relationship]:
    return {
        Relationship(source, attributes["relation"], target)
        for source, target, attributes in graph.edges(data=True)
    }
