"""Re-evaluate saved Torch checkpoints and existing reliability reports without retraining."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tbi_mlops.evaluation.research import run_research_evaluation


if __name__ == "__main__":
    report = run_research_evaluation()
    print("[research evaluation] synthetic-only evaluation complete")
    print(f"  dataset: {report['dataset_version']}")
    print("  JSON: artifacts/reports/research_evaluation.json")
    print("  Markdown: artifacts/reports/RESEARCH_EVALUATION.md")