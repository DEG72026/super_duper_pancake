# Image for all steps of the training pipeline (version 3).
# Everything the steps need is installed once here, so the steps on
# Vertex AI don't have to pip install anything when they start.
# Cloud Build builds this image and pushes it to Artifact Registry
# (see cloudbuild-train.yaml in pipeline/README.md).
FROM python:3.10-slim

# Same versions as PACKAGES in iris_pipeline.py and as the API.
# kfp must be the same version as the one that compiles the pipeline
# (pipeline/requirements.txt), because the steps use it to run.
RUN pip install --no-cache-dir \
    kfp==2.17.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.1.post1 \
    joblib==1.3.2
