"""Run the whole Iris pipeline on your own computer, without Google Cloud.

Every step runs as on Vertex AI, but locally: each step gets its own Python
environment with the packages from iris_pipeline.py. Uses data/iris.csv and
writes the model to local_run/models/ (not committed to Git).

Needs Python 3.10, 3.11 or 3.12 (scikit-learn 1.4.1 does not support 3.13).

Usage (from the repository root):
    pip install -r pipeline/requirements.txt
    python pipeline/run_local.py

Takes about 5-10 minutes, mostly installing packages for each step.
"""

from pathlib import Path

from kfp import local

from iris_pipeline import iris_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "local_run"

if __name__ == "__main__":
    local.init(
        runner=local.SubprocessRunner(use_venv=True),
        pipeline_root=str(OUTPUT_DIR / "pipeline_root"),
    )
    iris_pipeline(
        data_path=str(REPO_ROOT / "data" / "iris.csv"),
        model_dir=str(OUTPUT_DIR / "models"),
    )
    print(f"\nDone. Model saved to {OUTPUT_DIR / 'models' / 'model.joblib'}")
