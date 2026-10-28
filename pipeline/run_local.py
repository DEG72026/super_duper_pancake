"""Run the whole Iris pipeline on your own computer, without Google Cloud.

Uses data/iris.csv and writes the model to local_run/models/ (not in Git).
Needs Python 3.10, 3.11 or 3.12 (scikit-learn 1.4.1 does not support 3.13).

From the repository root:

    pip install -r pipeline/requirements.txt
    python pipeline/run_local.py

    Like Vertex AI without our image: every step gets a fresh Python
    environment and installs its packages first. Takes 5-10 minutes.

    pip install -r tests/requirements.txt
    python pipeline/run_local.py --image-mode

    Like Vertex AI WITH our image (version 3): steps install nothing and
    only use the packages already in your environment, which are the same
    as in pipeline/Dockerfile. If a step needed something that is not in
    the image, this run fails. Takes a few seconds.
"""

import argparse
import os
from pathlib import Path

from kfp import local

parser = argparse.ArgumentParser(description="Run the Iris pipeline locally")
parser.add_argument("--image-mode", action="store_true",
                    help="run the steps as in the pipeline image (nothing installed per step)")
args = parser.parse_args()

if args.image_mode:
    # any name works: locally the steps run in this Python environment,
    # the name only switches the pipeline to "no installing" mode
    os.environ["PIPELINE_IMAGE"] = "iris-pipeline:local-check"
else:
    os.environ.pop("PIPELINE_IMAGE", None)

from iris_pipeline import STEP_SETTINGS, iris_pipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "local_run"

if __name__ == "__main__":
    if args.image_mode:
        print("Image mode: steps install nothing, using", STEP_SETTINGS["base_image"])
    local.init(
        # use_venv=True: fresh environment per step (packages get installed)
        # use_venv=False: steps run in this environment, like inside our image
        runner=local.SubprocessRunner(use_venv=not args.image_mode),
        pipeline_root=str(OUTPUT_DIR / "pipeline_root"),
    )
    iris_pipeline(
        data_path=str(REPO_ROOT / "data" / "iris.csv"),
        model_dir=str(OUTPUT_DIR / "models"),
    )
    print(f"\nDone. Model saved to {OUTPUT_DIR / 'models' / 'model.joblib'}")
