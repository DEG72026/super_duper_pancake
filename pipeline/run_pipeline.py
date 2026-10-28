"""Send the compiled pipeline (YAML) to Vertex AI and start a run.

Cloud Build calls this script (cloudbuild/cloudbuild-train.yaml). You can
also run it yourself, e.g. from Cloud Shell:

    python pipeline/run_pipeline.py --project <PROJECT_ID>

Before starting the run, the YAML is also copied to a bucket folder that
acts as our pipeline repository (like in Lab 4), so we keep a copy of every
pipeline version that was actually run.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform, storage

DEFAULT_TEMPLATE = Path(__file__).parent / "iris_decision_tree_pipeline.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Iris pipeline on Vertex AI")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID"),
                        help="Google Cloud project ID")
    parser.add_argument("--region", default=os.getenv("REGION", "europe-west4"))
    parser.add_argument("--pipeline-root",
                        default=os.getenv("PIPELINE_ROOT", "gs://group7-iris-models/pipeline_root"),
                        help="where Vertex AI keeps the files of each run")
    parser.add_argument("--pipeline-repo",
                        default=os.getenv("PIPELINE_REPO", "gs://group7-iris-models/pipelines"),
                        help="bucket folder where we keep the compiled pipelines")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                        help="compiled pipeline YAML")
    parser.add_argument("--service-account", default=os.getenv("PIPELINE_SERVICE_ACCOUNT"),
                        help="service account the pipeline runs as (optional)")
    parser.add_argument("--enable-caching", action="store_true",
                        help="reuse results of unchanged steps (off by default so "
                             "every run really retrains)")
    parser.add_argument("--wait", action="store_true",
                        help="wait until the pipeline has finished")
    return parser.parse_args()


def upload_pipeline(template, pipeline_repo, project, run_time):
    """Copy the YAML to the pipeline repository bucket and return its gs:// path."""
    bucket_name, _, folder = pipeline_repo.removeprefix("gs://").partition("/")
    blob_name = f"{folder.rstrip('/')}/iris_decision_tree_pipeline-{run_time}.yaml".lstrip("/")
    storage.Client(project=project).bucket(bucket_name).blob(blob_name).upload_from_filename(template)
    return f"gs://{bucket_name}/{blob_name}"


def main():
    args = parse_args()
    if not args.project:
        raise SystemExit("Give the project ID with --project or the PROJECT_ID variable")

    run_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    uploaded = upload_pipeline(args.template, args.pipeline_repo, args.project, run_time)
    print(f"Pipeline YAML stored at {uploaded}")

    aiplatform.init(project=args.project, location=args.region)
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
    print(f"Started pipeline run: {job.resource_name}")


if __name__ == "__main__":
    main()
