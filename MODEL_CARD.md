# IdeaGrade GNN Model Card

## Model summary

IdeaGrade is a graph-regression model that estimates how closely a short
student explanation matches a teacher reference answer. It is a second opinion
inside an explainable grading workflow, not an autonomous grader.

| Property | Value |
| --- | --- |
| Model type | Message-passing graph neural network |
| Framework | PyTorch |
| Input | A comparison graph built from two directed idea graphs |
| Output | A score from 0 to 100 |
| Training examples | 834 |
| Feature version | 2 |
| Checkpoint | `models/idea_grade_gnn.pt` |

## Graph construction

spaCy dependency parsing extracts normalized subject–relation–object claims.
Each answer becomes a directed NetworkX graph. The model receives a combined
comparison graph containing:

- concept and relationship nodes from both answers;
- directed structural edges;
- cross-answer concept-match edges;
- lexical and WordNet-derived semantic compatibility features;
- explicit contradiction and negation signals; and
- graph-level coverage and topology summaries.

The network uses a 64-dimensional encoder, three message-passing layers,
mean graph pooling, and a regression head. Training uses AdamW and Smooth L1
loss with a fixed random seed.

## Training data

The included development corpus contains 834 synthetic rubric-labelled answer
pairs across 18 school-level topics. It covers exact answers, paraphrases,
omissions, reversed relationships, contradictions, unrelated answers, and
different answer lengths.

- `data/synthetic_training.jsonl`: 792 reproducibly generated examples.
- `data/starter_training.jsonl`: 42 targeted synthetic edge cases.

The corpus is designed to test the engineering pipeline. It is not evidence of
classroom validity, demographic fairness, or performance on authentic student
writing.

## Evaluation

The split is topic-held-out: entire topics are excluded from model fitting,
which is stricter than randomly splitting near-duplicate variants from every
topic.

| Metric | Result |
| --- | ---: |
| Validation MAE | 4.27 points |
| Validation RMSE | 10.38 points |
| Final all-data training MAE | 0.33 points |
| Held-out topics | climate change, photosynthesis, plate tectonics |

The large gap between training and validation error is an explicit warning
that the model can overfit synthetic rubric patterns. MAE also hides
variant-specific failures, so the training script reports the worst-performing
answer transformations.

## Intended use

- Demonstrating graph-based NLP and explainable ML.
- Comparing short factual explanations against a teacher-authored reference.
- Helping educators inspect missing, extra, or contradictory relationships.
- Research prototyping with anonymized, consented teacher labels.

## Out-of-scope use

- Autonomous final grades or high-stakes academic decisions.
- Ranking students, admissions screening, or disciplinary decisions.
- Long essays, creative writing, multilingual grading, or implicit reasoning.
- Claims of fairness or generalization beyond the evaluated synthetic corpus.

## Limitations

- Dependency parsing is sensitive to grammar, sentence complexity, and domain
  vocabulary.
- A valid alternative explanation may use concepts absent from the reference.
- WordNet coverage is uneven and does not establish contextual equivalence.
- The dataset is synthetic and small by modern ML standards.
- A single reference answer cannot represent every acceptable line of reasoning.
- The current uncertainty display uses held-out MAE, not calibrated
  per-example predictive uncertainty.

## Responsible deployment

Keep a teacher in the decision loop, display extracted evidence alongside both
scores, allow educators to override predictions, and log disagreements for
analysis. Before classroom use, collect anonymized data from the intended
subject and grade level, create a genuinely untouched test set, evaluate
subgroups, document annotation agreement, and obtain appropriate institutional
and parental approvals.

## Reproducibility

```bash
python generate_training_data.py
python train_gnn.py
python -m unittest discover -s tests -v
```

Training metadata is stored inside the checkpoint and can be inspected with
PyTorch. The generator and trainer both use fixed seeds.
