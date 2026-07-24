"""Display helpers for the command-line and Streamlit interfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from graph_builder import build_idea_graph


def graph_as_edges(text: str) -> list[str]:
    """Return labelled graph edges in a human-readable form."""
    graph = build_idea_graph(text)
    return [
        f"{source} --{data['relation']}--> {target}"
        for source, target, data in graph.edges(data=True)
    ]


def draw_graph(graph: nx.DiGraph, title: str, dark: bool = False):
    """Render an idea graph as a Matplotlib figure for Streamlit."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    text_color = "#E5E7EB" if dark else "#111827"
    node_color = "#27345A" if dark else "#DBEAFE"
    node_edge = "#8B9DFF" if dark else "#2563EB"
    relation_color = "#C4B5FD" if dark else "#7C3AED"
    edge_color = "#94A3B8" if dark else "#64748B"
    axis.set_title(title, fontweight="bold", pad=14, color=text_color)
    axis.axis("off")
    if not graph.nodes:
        axis.text(
            0.5,
            0.5,
            "No explicit relationships found",
            ha="center",
            va="center",
            color=text_color,
        )
        return figure

    positions = nx.spring_layout(graph, seed=21, k=1.5)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_color,
        edgecolors=node_edge,
        node_size=2600,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=10,
        font_weight="bold",
        font_color=text_color,
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph, positions, edge_color=edge_color, arrows=True, arrowsize=20,
        connectionstyle="arc3,rad=0.08", width=1.8, ax=axis
    )
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=nx.get_edge_attributes(graph, "relation"),
        font_color=relation_color, font_size=9, rotate=False, ax=axis
    )
    figure.tight_layout()
    return figure
