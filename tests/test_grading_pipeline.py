"""Regression tests for the explainable and learned grading paths."""

from __future__ import annotations

import json
import unittest

from analysis_service import analyze_answers
from data_store import load_teacher_scores
from gnn_model import load_trained_model
from grader import grade_answer
from settings import DATA_DIR, MODEL_PATH


class StructuralGradingTests(unittest.TestCase):
    def test_exact_answer_receives_full_structural_score(self) -> None:
        answer = "Plants use sunlight. Plants produce glucose."

        result = grade_answer(answer, answer)

        self.assertEqual(result.score, 100.0)
        self.assertEqual(result.missing_relationships, ())
        self.assertEqual(result.extra_relationships, ())

    def test_missing_claim_reduces_score_and_is_explained(self) -> None:
        reference = "Plants use sunlight. Plants produce glucose."
        student = "Plants use sunlight."

        result = grade_answer(reference, student)

        self.assertLess(result.score, 100.0)
        self.assertTrue(
            any("produce" in relationship for relationship in result.missing_relationships)
        )


class DatasetAndModelTests(unittest.TestCase):
    def test_published_training_corpus_matches_checkpoint(self) -> None:
        records = load_teacher_scores()
        model = load_trained_model()

        self.assertIsNotNone(model)
        self.assertEqual(len(records), 834)
        self.assertEqual(model.example_count, len(records))
        self.assertAlmostEqual(model.validation_mae, 4.27, places=2)

    def test_every_published_record_has_required_fields(self) -> None:
        paths = sorted(DATA_DIR.glob("*training.jsonl"))
        self.assertGreaterEqual(len(paths), 2)

        for path in paths:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                record = json.loads(line)
                with self.subTest(path=path.name, line=line_number):
                    self.assertIn("reference", record)
                    self.assertIn("student", record)
                    self.assertIn("teacher_score", record)
                    self.assertGreaterEqual(float(record["teacher_score"]), 0.0)
                    self.assertLessEqual(float(record["teacher_score"]), 100.0)

    def test_gnn_prediction_is_bounded(self) -> None:
        model = load_trained_model()
        self.assertIsNotNone(model)

        analysis = analyze_answers(
            "Plants use sunlight to produce glucose.",
            "Plants use sunlight to produce glucose.",
            model,
        )

        self.assertIsNotNone(analysis.gnn_score)
        self.assertGreaterEqual(analysis.gnn_score, 0.0)
        self.assertLessEqual(analysis.gnn_score, 100.0)
        self.assertTrue(MODEL_PATH.exists())


if __name__ == "__main__":
    unittest.main()
