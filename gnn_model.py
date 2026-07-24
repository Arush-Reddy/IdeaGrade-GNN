"""A small, dependency-light graph neural network for score regression.

The comparison graph contains both the reference and student idea graphs. Exact
concept matches form cross-graph edges; relationship nodes preserve verb labels.
This lets message passing learn patterns from teacher-provided scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

import torch
from torch import nn
import nltk
from nltk.corpus import wordnet

from graph_builder import build_idea_graph, graph_relationships
from extractor import get_nlp
from settings import MODEL_PATH, NLTK_DATA_PATH


SEMANTIC_SIZE = 16
FEATURE_SIZE = 9 + SEMANTIC_SIZE
SUMMARY_SIZE = 9
nltk.data.path.insert(0, str(NLTK_DATA_PATH))


@lru_cache(maxsize=512)
def _semantic_features(text: str) -> tuple[float, ...]:
    """Return compact contextual spaCy features for a concept or relation."""
    doc = get_nlp()(text.replace("not_", "not "))
    if not len(doc) or doc.tensor.size == 0:
        return (0.0,) * SEMANTIC_SIZE
    values = doc.tensor.mean(axis=0).tolist()[:SEMANTIC_SIZE]
    values += [0.0] * (SEMANTIC_SIZE - len(values))
    magnitude = sqrt(sum(float(value) ** 2 for value in values)) or 1.0
    return tuple(float(value) / magnitude for value in values)


def _semantic_similarity(left: str, right: str) -> float:
    left_values = _semantic_features(left)
    right_values = _semantic_features(right)
    cosine = sum(a * b for a, b in zip(left_values, right_values))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


@lru_cache(maxsize=256)
def _wordnet_relations(word: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return WordNet synonyms and antonyms for a normalized verb."""
    normalized = word.removeprefix("not_").replace("_", " ")
    synonyms: set[str] = {normalized}
    antonyms: set[str] = set()
    try:
        for synset in wordnet.synsets(normalized, pos=wordnet.VERB):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
                antonyms.update(
                    antonym.name().replace("_", " ") for antonym in lemma.antonyms()
                )
    except LookupError:
        pass
    return frozenset(synonyms), frozenset(antonyms)


def _relation_compatibility(reference: str, student: str) -> tuple[float, float]:
    """Return semantic agreement and explicit contradiction signals."""
    if reference == student:
        return 1.0, 0.0
    reference_negated = reference.startswith("not_")
    student_negated = student.startswith("not_")
    if reference_negated != student_negated:
        return 0.0, 1.0
    reference_word = reference.removeprefix("not_")
    student_word = student.removeprefix("not_")
    reference_synonyms, reference_antonyms = _wordnet_relations(reference_word)
    student_synonyms, student_antonyms = _wordnet_relations(student_word)
    if student_word in reference_antonyms or reference_word in student_antonyms:
        return 0.0, 1.0
    if student_word in reference_synonyms or reference_word in student_synonyms:
        return 0.95, 0.0
    return 0.45 * _semantic_similarity(reference_word, student_word), 0.0


@dataclass
class GraphSample:
    features: torch.Tensor
    edges: torch.Tensor
    summary: torch.Tensor
    score: torch.Tensor | None = None


def make_comparison_graph(reference_text: str, student_text: str, score: float | None = None) -> GraphSample:
    """Build a single graph that represents an answer comparison."""
    reference = graph_relationships(build_idea_graph(reference_text))
    student = graph_relationships(build_idea_graph(student_text))
    reference_concepts = {part for edge in reference for part in (edge.subject, edge.object)}
    student_concepts = {part for edge in student for part in (edge.subject, edge.object)}
    shared_concepts = reference_concepts & student_concepts
    shared_relationships = reference & student

    features: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    index: dict[str, int] = {}

    def add_node(identifier: str, values: list[float], semantic_text: str = "") -> int:
        if identifier not in index:
            index[identifier] = len(features)
            features.append(values + list(_semantic_features(semantic_text)))
        return index[identifier]

    def add_answer_graph(relationships, prefix: str, is_reference: float) -> None:
        for number, relation in enumerate(sorted(relationships)):
            subject = add_node(
                f"{prefix}:concept:{relation.subject}",
                [is_reference, 1.0 - is_reference, 0.0, 0.0, float(relation.subject in shared_concepts), 0.0, 0.0, 0.0, 0.0],
                relation.subject,
            )
            target = add_node(
                f"{prefix}:concept:{relation.object}",
                [is_reference, 1.0 - is_reference, 0.0, 0.0, float(relation.object in shared_concepts), 0.0, 0.0, 0.0, 0.0],
                relation.object,
            )
            relation_node = add_node(
                f"{prefix}:relationship:{number}",
                [
                    is_reference,
                    1.0 - is_reference,
                    1.0,
                    0.0,
                    float(relation in shared_relationships),
                    float(relation.relation.startswith("not_")),
                    0.0,
                    0.0,
                    0.0,
                ],
                relation.relation,
            )
            edges.extend([(subject, relation_node), (relation_node, target)])

    add_answer_graph(reference, "reference", 1.0)
    add_answer_graph(student, "student", 0.0)

    for concept in shared_concepts:
        left = index[f"reference:concept:{concept}"]
        right = index[f"student:concept:{concept}"]
        edges.extend([(left, right), (right, left)])

    reference_connections = {(edge.subject, edge.object) for edge in reference}
    student_connections = {(edge.subject, edge.object) for edge in student}
    concept_recall = len(shared_concepts) / len(reference_concepts) if reference_concepts else 0.0
    relationship_recall = len(shared_relationships) / len(reference) if reference else 0.0
    structure_recall = (
        len(reference_connections & student_connections) / len(reference_connections)
        if reference_connections else 0.0
    )
    semantic_matches = []
    contradiction_matches = []
    for reference_edge in reference:
        same_connection = [
            student_edge
            for student_edge in student
            if (
                student_edge.subject == reference_edge.subject
                and student_edge.object == reference_edge.object
            )
        ]
        compatibility = [
            _relation_compatibility(reference_edge.relation, student_edge.relation)
            for student_edge in same_connection
        ]
        semantic_matches.append(max((item[0] for item in compatibility), default=0.0))
        contradiction_matches.append(max((item[1] for item in compatibility), default=0.0))
    relation_semantic_recall = (
        sum(semantic_matches) / len(semantic_matches) if semantic_matches else 0.0
    )
    contradiction_ratio = (
        sum(contradiction_matches) / len(reference) if reference else 0.0
    )
    negation_ratio = (
        sum(edge.relation.startswith("not_") for edge in student) / len(student)
        if student else 0.0
    )
    reverse_ratio = (
        sum((target, source) in student_connections for source, target in reference_connections)
        / len(reference_connections)
        if reference_connections else 0.0
    )
    extra_ratio = min(
        max(len(student) - len(reference), 0) / max(len(reference), 1),
        1.0,
    )
    length_ratio = min(len(student) / max(len(reference), 1), 2.0) / 2.0
    summary = add_node(
        "comparison:summary",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, concept_recall, relationship_recall, structure_recall],
        "",
    )
    for node in range(summary):
        edges.extend([(summary, node), (node, summary)])

    # Bidirectional messages capture context from both source and target claims.
    edges = edges + [(target, source) for source, target in edges]
    if not features:
        features = [[0.0] * FEATURE_SIZE]
    if not edges:
        edges = [(0, 0)]
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    degrees = torch.zeros(feature_tensor.size(0))
    for source, target in edges:
        degrees[source] += 1
        degrees[target] += 1
    feature_tensor[:, 3] = degrees / max(float(degrees.max()), 1.0)
    edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    summary_tensor = torch.tensor(
        [
            concept_recall,
            relationship_recall,
            structure_recall,
            relation_semantic_recall,
            negation_ratio,
            reverse_ratio,
            extra_ratio,
            length_ratio,
            contradiction_ratio,
        ],
        dtype=torch.float32,
    )
    score_tensor = torch.tensor([score / 100.0], dtype=torch.float32) if score is not None else None
    return GraphSample(feature_tensor, edge_tensor, summary_tensor, score_tensor)


def batch_samples(
    samples: list[GraphSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine variable-size graphs into one batch without PyTorch Geometric."""
    feature_blocks, edge_blocks, batch, summaries, labels = [], [], [], [], []
    offset = 0
    for graph_number, sample in enumerate(samples):
        feature_blocks.append(sample.features)
        edge_blocks.append(sample.edges + offset)
        batch.extend([graph_number] * sample.features.size(0))
        summaries.append(sample.summary)
        labels.append(sample.score)
        offset += sample.features.size(0)
    return (
        torch.cat(feature_blocks),
        torch.cat(edge_blocks, dim=1),
        torch.tensor(batch, dtype=torch.long),
        torch.stack(summaries),
        torch.cat(labels),
    )


class MessageLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(), nn.LayerNorm(width))

    def forward(self, values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        source, target = edges
        messages = torch.zeros_like(values)
        messages.index_add_(0, target, values[source])
        counts = torch.bincount(target, minlength=values.size(0)).clamp(min=1).unsqueeze(1)
        return self.update(torch.cat([values, messages / counts], dim=1))


class IdeaGraphRegressor(nn.Module):
    def __init__(self, width: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_SIZE, width), nn.ReLU())
        self.layers = nn.ModuleList([MessageLayer(width), MessageLayer(width), MessageLayer(width)])
        self.head = nn.Sequential(
            nn.Linear(width + SUMMARY_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        summaries: torch.Tensor,
    ) -> torch.Tensor:
        values = self.encoder(features)
        for layer in self.layers:
            values = values + layer(values, edges)
        graph_count = int(batch.max().item()) + 1
        pooled = torch.zeros(graph_count, values.size(1), device=values.device)
        pooled.index_add_(0, batch, values)
        counts = torch.bincount(batch, minlength=graph_count).clamp(min=1).unsqueeze(1)
        return self.head(torch.cat([pooled / counts, summaries], dim=1)).squeeze(1)


def load_trained_model() -> IdeaGraphRegressor | None:
    """Return a trained model when one exists locally, otherwise None."""
    if not MODEL_PATH.exists():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = IdeaGraphRegressor(width=int(checkpoint.get("width", 48)))
    model.load_state_dict(checkpoint["state_dict"])
    model.validation_mae = float(checkpoint.get("validation_mae", 0.0))
    model.example_count = int(checkpoint.get("example_count", 0))
    model.eval()
    return model


def predict_score(model: IdeaGraphRegressor, reference: str, student: str) -> float:
    sample = make_comparison_graph(reference, student)
    batch = torch.zeros(sample.features.size(0), dtype=torch.long)
    with torch.no_grad():
        return round(
            float(
                model(
                    sample.features,
                    sample.edges,
                    batch,
                    sample.summary.unsqueeze(0),
                ).item()
                * 100
            ),
            1,
        )
