"""Train and evaluate the graph-regression model with a topic-held-out split."""

from __future__ import annotations

import random

import torch
from torch import nn

from data_store import load_teacher_scores
from gnn_model import MODEL_PATH, IdeaGraphRegressor, batch_samples, make_comparison_graph
from graph_builder import build_idea_graph


MINIMUM_EXAMPLES = 100
EPOCHS = 700


def _fit(samples, epochs: int = EPOCHS) -> IdeaGraphRegressor:
    torch.manual_seed(21)
    model = IdeaGraphRegressor(width=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0002)
    loss_function = nn.SmoothL1Loss(beta=0.08)
    features, edges, batch, summaries, labels = batch_samples(samples)
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        predictions = model(features, edges, batch, summaries)
        loss = loss_function(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def _evaluate(model, samples) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        features, edges, batch, summaries, labels = batch_samples(samples)
        predictions = model(features, edges, batch, summaries)
        errors = torch.abs(predictions - labels) * 100
        return float(errors.mean()), float(torch.sqrt(torch.mean((predictions - labels) ** 2)) * 100)


def _variant_diagnostics(model, records, samples) -> list[tuple[str, float]]:
    model.eval()
    with torch.no_grad():
        features, edges, batch, summaries, labels = batch_samples(samples)
        errors = (torch.abs(model(features, edges, batch, summaries) - labels) * 100).tolist()
    grouped: dict[str, list[float]] = {}
    for record, error in zip(records, errors):
        grouped.setdefault(record.get("variant", "unlabelled"), []).append(error)
    return sorted(
        ((name, sum(values) / len(values)) for name, values in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def main() -> None:
    records = load_teacher_scores()
    if len(records) < MINIMUM_EXAMPLES:
        raise SystemExit(
            f"Need at least {MINIMUM_EXAMPLES} examples; found {len(records)}. "
            "Run python generate_training_data.py or collect teacher scores in the app."
        )

    invalid_records = [
        record
        for record in records
        if not build_idea_graph(record["reference"]).edges
        or not build_idea_graph(record["student"]).edges
    ]
    invalid_generated = [
        record for record in invalid_records if record.get("source") == "synthetic_rubric_v2"
    ]
    if invalid_generated:
        raise SystemExit(
            f"Dataset integrity check failed: {len(invalid_generated)} generated examples have empty graphs. "
            "Regenerate or correct the training data before training."
        )
    if invalid_records:
        invalid_ids = {id(record) for record in invalid_records}
        records = [record for record in records if id(record) not in invalid_ids]
        print(f"Skipped {len(invalid_records)} legacy examples with empty graphs.")

    topics = sorted({record["topic"] for record in records if record.get("topic")})
    random.Random(21).shuffle(topics)
    validation_topics = set(topics[: max(2, len(topics) // 5)])
    training_records = [
        record for record in records if record.get("topic") not in validation_topics
    ]
    validation_records = [
        record for record in records if record.get("topic") in validation_topics
    ]
    training = [
        make_comparison_graph(record["reference"], record["student"], float(record["teacher_score"]))
        for record in training_records
    ]
    validation = [
        make_comparison_graph(record["reference"], record["student"], float(record["teacher_score"]))
        for record in validation_records
    ]

    evaluation_model = _fit(training)
    mae, rmse = _evaluate(evaluation_model, validation)
    diagnostics = _variant_diagnostics(evaluation_model, validation_records, validation)

    # Once the held-out metric is recorded, train the deployable model on all
    # available data so no teacher-labelled example is wasted.
    all_samples = training + validation
    final_model = _fit(all_samples, epochs=EPOCHS)
    training_mae, _ = _evaluate(final_model, all_samples)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "validation_mae": round(mae, 2),
            "validation_rmse": round(rmse, 2),
            "training_mae": round(training_mae, 2),
            "example_count": len(records),
            "validation_topics": sorted(validation_topics),
            "feature_version": 2,
            "width": 64,
        },
        MODEL_PATH,
    )
    print(f"Examples: {len(records)}")
    print(f"Held-out topics: {', '.join(sorted(validation_topics))}")
    print(f"Validation MAE: {mae:.2f} points")
    print(f"Validation RMSE: {rmse:.2f} points")
    print(
        "Worst validation variants: "
        + ", ".join(f"{name}={error:.1f}" for name, error in diagnostics[:5])
    )
    print(f"Final all-data training MAE: {training_mae:.2f} points")
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
