"""Explainable graph-based scoring for a reference and student answer."""

from __future__ import annotations

from dataclasses import dataclass

from graph_builder import build_idea_graph, graph_relationships


def _percent(value: float) -> float:
    return round(value * 100, 1)


@dataclass(frozen=True)
class GradeResult:
    score: float
    concept_score: float
    relationship_score: float
    structure_score: float
    matched_concepts: tuple[str, ...]
    missing_relationships: tuple[str, ...]
    extra_relationships: tuple[str, ...]

    def to_report(self) -> str:
        def display(items: tuple[str, ...]) -> str:
            return "\n  - " + "\n  - ".join(items) if items else " none"

        return (
            "\nSTRUCTURAL GRADING REPORT\n"
            "=" * 28
            + f"\nOverall score: {self.score:.1f}/100"
            + f"\nConcept coverage: {self.concept_score:.1f}%"
            + f"\nRelationship accuracy: {self.relationship_score:.1f}%"
            + f"\nArgument structure: {self.structure_score:.1f}%"
            + f"\n\nMatched concepts:{display(self.matched_concepts)}"
            + f"\n\nMissing reference relationships:{display(self.missing_relationships)}"
            + f"\n\nAdditional student relationships:{display(self.extra_relationships)}"
        )


def _format_relationship(triple) -> str:
    return f"{triple.subject} --{triple.relation}--> {triple.object}"


def grade_answer(reference_text: str, student_text: str) -> GradeResult:
    """Grade structural overlap, returning feedback rather than only one number.

    Weighting: 30% concept coverage, 45% correct relationships, 25% graph structure.
    A teacher should treat this as a review aid and set the final grade themselves.
    """
    reference = build_idea_graph(reference_text)
    student = build_idea_graph(student_text)
    reference_nodes, student_nodes = set(reference.nodes), set(student.nodes)
    reference_edges, student_edges = graph_relationships(reference), graph_relationships(student)

    matched_nodes = reference_nodes & student_nodes
    matched_edges = reference_edges & student_edges

    concept_score = len(matched_nodes) / len(reference_nodes) if reference_nodes else 0.0
    relationship_score = len(matched_edges) / len(reference_edges) if reference_edges else 0.0

    # Structure measures whether concepts are connected in the intended direction,
    # independent of verb choice. Relationship accuracy above remains stricter.
    reference_connections = {(edge.subject, edge.object) for edge in reference_edges}
    student_connections = {(edge.subject, edge.object) for edge in student_edges}
    structure_score = (
        len(reference_connections & student_connections) / len(reference_connections)
        if reference_connections else 0.0
    )
    total = 0.30 * concept_score + 0.45 * relationship_score + 0.25 * structure_score

    return GradeResult(
        score=_percent(total),
        concept_score=_percent(concept_score),
        relationship_score=_percent(relationship_score),
        structure_score=_percent(structure_score),
        matched_concepts=tuple(sorted(matched_nodes)),
        missing_relationships=tuple(_format_relationship(edge) for edge in sorted(reference_edges - student_edges)),
        extra_relationships=tuple(_format_relationship(edge) for edge in sorted(student_edges - reference_edges)),
    )
