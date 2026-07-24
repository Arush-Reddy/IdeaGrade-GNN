"""Optional command-line interface for structural grading."""

from __future__ import annotations

import argparse

from grader import grade_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare student and teacher idea graphs.")
    parser.add_argument("--reference", required=True, help="Teacher reference answer")
    parser.add_argument("--student", required=True, help="Student answer")
    args = parser.parse_args()
    print(grade_answer(args.reference, args.student).to_report())


if __name__ == "__main__":
    main()
