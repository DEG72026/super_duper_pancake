"""Compile the Iris training pipeline into a YAML file that Vertex AI can run.

Usage (from the repository root):
    python pipeline/compile_pipeline.py

With PIPELINE_IMAGE set (Cloud Build does this), all steps use that image.
The YAML in Git is compiled without it, so it works anywhere.
"""

from pathlib import Path

from kfp import compiler

from iris_pipeline import PIPELINE_IMAGE, iris_pipeline

OUTPUT_FILE = Path(__file__).parent / "iris_decision_tree_pipeline.yaml"

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=str(OUTPUT_FILE),
    )
    print(f"Pipeline compiled to {OUTPUT_FILE}")
    print(f"Steps run in: {PIPELINE_IMAGE or 'python:3.10-slim (installing packages per step)'}")
