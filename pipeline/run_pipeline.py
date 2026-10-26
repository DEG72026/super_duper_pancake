"""Submit the compiled Iris pipeline (YAML) to Vertex AI Pipelines.

Used by Cloud Build (cloudbuild/cloudbuild-train.yaml) for continuous
training, and can also be run by hand.

Usage (from the repository root):
    python pipeline/run_pipeline.py --project <PROJECT_ID>

Settings can also come from environment variables:
    PROJECT_ID, REGION, PIPELINE_ROOT
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform

DEFAULT_TEMPLATE = Path(__file__).parent / "iris_decision_tree_pipeline.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Iris pipeline on Vertex AI.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID"),
                        help="Google Cloud project ID")
    parser.add_argument("--region", default=os.getenv("REGION", "europe-west4"))
    parser.add_argument("--pipeline-root",
                        default=os.getenv("PIPELINE_ROOT", "gs://group7-iris-models/pipeline_root"),
                        help="Bucket folder where Vertex AI stores run files")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                        help="Compiled pipeline YAML")
    parser.add_argument("--service-account", default=os.getenv("PIPELINE_SERVICE_ACCOUNT"),
                        help="Service account the pipeline runs as (optional)")
    parser.add_argument("--enable-caching", action="store_true",
                        help="Reuse results of unchanged steps (off by default, so "
                             "every run really retrains)")
    parser.add_argument("--wait", action="store_true",
                        help="Wait until the pipeline finishes instead of returning")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.project:
        raise SystemExit("Set the project ID with --project or the PROJECT_ID variable.")

    aiplatform.init(project=args.project, location=args.region)

    run_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = aiplatform.PipelineJob(
        display_name=f"iris-decision-tree-training-{run_time}",
        template_path=args.template,
        pipeline_root=args.pipeline_root,
        enable_caching=args.enable_caching,
    )

    if args.wait:
        job.run(service_account=args.service_account)
    else:
        job.submit(service_account=args.service_account)
    print(f"Submitted pipeline job: {job.resource_name}")


if __name__ == "__main__":
    main()
