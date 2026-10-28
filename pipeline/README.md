# Vertex AI training pipeline (Pranav)

Turns Renske's local Decision Tree script (`DecisionTreeModel.py`) into a
Vertex AI pipeline. It trains four trees, picks the best one, checks it
against the model that's currently in use, and only then saves it for the API.

## Steps

```
load_data → split_data → train_model ×4 (depth 2, 3, 4, None; in parallel)
          → evaluate_model ×4 → select_best_model → check_model
          → save_model (only if check_model says "deploy")
```

| Step | Input | Output |
|---|---|---|
| `load_data` | `gs://group7-iris-data/iris.csv` | checked dataset |
| `split_data` | dataset | train set (80%) and test set (20%), stratified, seed 42 |
| `train_model` | train set, max_depth | trained Decision Tree |
| `evaluate_model` | model, test set | accuracy (visible in the Vertex AI UI) |
| `select_best_model` | 4 models + 4 accuracies | best model; on a tie the simpler tree |
| `check_model` | best model, current model's accuracy | `"deploy"` or `"keep"` |
| `save_model` | best model | `model/model.joblib`, `model/metrics.json`, `model/versions/model-<time>.joblib` |

### Quality gate (champion vs challenger, Lab 4)

`check_model` compares the new model (challenger) with the one in use
(champion, whose accuracy is in `model/metrics.json`). The new model is only
saved if:

- accuracy ≥ `min_accuracy` (pipeline parameter, default 0.9)
- accuracy ≥ the champion's accuracy
- it predicts three known flowers correctly (one per species)

Otherwise the pipeline still finishes successfully, but `save_model` is skipped
and the API keeps the old model. The first run has no champion, so a good
model is always saved.

## Agreements with the rest of the team

- **API (Himey):** the current model is always at `gs://group7-iris-models/model/model.joblib`
  (joblib, scikit-learn 1.4.1.post1, labels 0 = setosa, 1 = versicolor, 2 = virginica,
  features in the order sepal_length, sepal_width, petal_length, petal_width).
  `model.joblib` is written last, so it's the file to watch for a Pub/Sub trigger.
- **Data:** `data/iris.csv` (150 rows, numeric `species`) must be in `gs://group7-iris-data/iris.csv`.
- **Cloud Build (Charlotte):** compile with `compile_pipeline.py`, start with `run_pipeline.py`.

## Files

| File | Purpose |
|---|---|
| `iris_pipeline.py` | the pipeline: all steps and how they connect |
| `compile_pipeline.py` | compiles the pipeline into `iris_decision_tree_pipeline.yaml` |
| `iris_decision_tree_pipeline.yaml` | the compiled pipeline Vertex AI runs (versioned in Git) |
| `run_pipeline.py` | copies the YAML to the pipeline repository bucket and starts a run on Vertex AI |
| `run_local.py` | runs the whole pipeline on your own computer (no Google Cloud) |
| `requirements.txt` | packages to compile and start the pipeline |
| `Dockerfile` | image with all packages for the pipeline steps (version 3) |
| `../tests/test_prediction.py` | tests for the steps, the quality gate and the model's predictions |

## Run it

From the repository root.

**Tests** (a few seconds, no cloud):
```bash
pip install -r tests/requirements.txt
pytest tests/
```

**Whole pipeline on your laptop** (5–10 minutes, saves to `local_run/`):
```bash
pip install -r pipeline/requirements.txt
python pipeline/compile_pipeline.py
python pipeline/run_local.py
```

**Same, but as inside the pipeline image** (a few seconds; steps install
nothing, so it fails if a step needs a package that isn't in `pipeline/Dockerfile`):
```bash
pip install -r tests/requirements.txt
python pipeline/run_local.py --image-mode
```

**Test the real image on Vertex AI by hand** (Cloud Shell, before the trigger exists):
```bash
IMAGE=europe-west4-docker.pkg.dev/<PROJECT_ID>/mlops-repo-group7/iris-pipeline:manual
gcloud builds submit pipeline/ --tag $IMAGE --region europe-west4
PIPELINE_IMAGE=$IMAGE python pipeline/compile_pipeline.py
python pipeline/run_pipeline.py --project <PROJECT_ID>
git checkout pipeline/iris_decision_tree_pipeline.yaml   # keep the Git YAML without the image
```

**On Vertex AI** (e.g. from Cloud Shell):
```bash
pip install -r pipeline/requirements.txt
gsutil cp data/iris.csv gs://group7-iris-data/iris.csv   # once
python pipeline/compile_pipeline.py
python pipeline/run_pipeline.py --project <PROJECT_ID>
```
Then watch the run under **Vertex AI → Pipelines**. Each run's YAML is also
stored in `gs://group7-iris-models/pipelines/` (our pipeline repository).

Caching is off by default so every run really retrains. `--enable-caching`
turns it on, `--wait` waits until the run is done.

**Whenever `iris_pipeline.py` changes, run `compile_pipeline.py` again and commit
the new YAML too.**

## Pipeline image (version 3)

All steps can run in one pre-built image (`pipeline/Dockerfile`) with pandas,
scikit-learn etc. already installed, so steps start right away instead of
installing packages first (in-line components with a custom image, Lab 4 slide 11).

- `PIPELINE_IMAGE` set when compiling → all steps use that image, nothing is installed
- not set (laptop, the YAML in Git) → `python:3.10-slim`, each step installs its packages

Cloud Build builds the image, pushes it to Artifact Registry and compiles the
pipeline with `PIPELINE_IMAGE` pointing to it. So CI/CD now also builds the
training pipeline component, as the assignment asks. Run locally, the whole
pipeline took 7 seconds in this mode instead of about 4 minutes.

The copy of each run's YAML in `gs://group7-iris-models/pipelines/` is the one
that was really used (with the image), the YAML in Git is the version without it.

## Suggested `cloudbuild/cloudbuild-train.yaml`

```yaml
substitutions:
  _REGION: europe-west4
  _AR_REPO: mlops-repo-group7
  _PIPELINE_ROOT: gs://group7-iris-models/pipeline_root

steps:
  # 1. test the pipeline code; the build stops here if a test fails
  - id: test
    name: 'python:3.10-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r tests/requirements.txt
        pytest tests/

  # 2. build the image for the pipeline steps
  - id: build-pipeline-image
    name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/iris-pipeline:${BUILD_ID}', 'pipeline']

  # 3. push it to Artifact Registry
  - id: push-pipeline-image
    name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/iris-pipeline:${BUILD_ID}']

  # 4. compile with that image, store the YAML and start the run on Vertex AI
  - id: compile-and-run
    name: 'python:3.10-slim'
    entrypoint: 'bash'
    env:
      - 'PIPELINE_IMAGE=${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/iris-pipeline:${BUILD_ID}'
    args:
      - '-c'
      - |
        pip install -r pipeline/requirements.txt
        python pipeline/compile_pipeline.py
        python pipeline/run_pipeline.py --project ${PROJECT_ID} --region ${_REGION} --pipeline-root ${_PIPELINE_ROOT}

timeout: '1800s'
options:
  logging: CLOUD_LOGGING_ONLY
```

Trigger: on pushes to `main` that change `pipeline/**`, `tests/**` or `data/**`.

Permissions:
- Cloud Build service account: **Vertex AI User**, **Service Account User**,
  **Artifact Registry Writer**, and write access to the model bucket (pipeline repository)
- the account the pipeline runs as (by default the Compute Engine default
  service account): **Artifact Registry Reader** (to pull the image), read access
  to the data bucket and write access to the model bucket

## Not done (ideas for the report)

- **Fully containerized components** (Lab 5): each step as its own script and
  image (`@dsl.container_component`), so steps can be tested and reused on
  their own. We use one shared image for in-line components instead.
- **Real hyperparameter tuning** instead of four fixed depths.
- **Scheduled retraining** (Cloud Scheduler) or retraining when new data arrives.
