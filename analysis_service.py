"""UI-independent orchestration for answer analysis."""

from __future__ import annotations

from dataclasses import dataclass

from gnn_model import IdeaGraphRegressor, predict_score
from grader import GradeResult, grade_answer
from graph_builder import build_idea_graph, graph_relationships
from utils import graph_as_edges


@dataclass(frozen=True)
class AnalysisBundle:
    reference: str
    student: str
    grade: GradeResult
    reference_edges: tuple[str, ...]
    student_edges: tuple[str, ...]
    gnn_score: float | None
    validation_mae: float | None
    training_examples: int


def relationship_rows(text: str) -> list[dict[str, str]]:
    relationships = sorted(graph_relationships(build_idea_graph(text)))
    return [
        {
            "Concept 1": relationship.subject,
            "Relationship": relationship.relation.replace("not_", "not "),
            "Concept 2": relationship.object,
        }
        for relationship in relationships
    ]


def analyze_answers(
    reference: str,
    student: str,
    model: IdeaGraphRegressor | None,
) -> AnalysisBundle:
    reference = reference.strip()
    student = student.strip()
    grade = grade_answer(reference, student)
    return AnalysisBundle(
        reference=reference,
        student=student,
        grade=grade,
        reference_edges=tuple(graph_as_edges(reference)),
        student_edges=tuple(graph_as_edges(student)),
        gnn_score=predict_score(model, reference, student) if model is not None else None,
        validation_mae=(
            float(getattr(model, "validation_mae", 0.0)) if model is not None else None
        ),
        training_examples=(
            int(getattr(model, "example_count", 0)) if model is not None else 0
        ),
    )
