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

## Suggested `cloudbuild/cloudbuild-train.yaml`

```yaml
substitutions:
  _REGION: europe-west4
  _PIPELINE_ROOT: gs://group7-iris-models/pipeline_root

steps:
  # test the pipeline code first; stop if something is broken
  - name: 'python:3.10-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r tests/requirements.txt
        pytest tests/

  # compile, store the YAML in the pipeline repository and start the run
  - name: 'python:3.10-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r pipeline/requirements.txt
        python pipeline/compile_pipeline.py
        python pipeline/run_pipeline.py --project $PROJECT_ID --region ${_REGION} --pipeline-root ${_PIPELINE_ROOT}

timeout: '1200s'
options:
  logging: CLOUD_LOGGING_ONLY
```

Permissions: the Cloud Build service account needs **Vertex AI User**,
**Service Account User** and write access to the model bucket (for the
pipeline repository). The account the pipeline runs as (by default the Compute
Engine default service account) needs read access to the data bucket and
write access to the model bucket.

## Not done (ideas for the report)

- **Own Docker image for the steps** (Lab 5): each step now installs its
  packages when it starts. A pre-built image would make steps start faster,
  but it's optional for the assignment and needs an extra Cloud Build step.
- **Real hyperparameter tuning** instead of four fixed depths.
- **Scheduled retraining** (Cloud Scheduler) or retraining when new data arrives.
