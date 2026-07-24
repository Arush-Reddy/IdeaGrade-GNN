"""Deployment-safe paths and application metadata."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "idea_grade_gnn.pt"
TEACHER_DATA_PATH = DATA_DIR / "teacher_calibrations.jsonl"
NLTK_DATA_PATH = DATA_DIR / "nltk_data"

APP_NAME = "IdeaGrade"
APP_VERSION = "1.0.0"
MINIMUM_TRAINING_EXAMPLES = 100
