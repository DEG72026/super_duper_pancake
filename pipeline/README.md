# Vertex AI training pipeline (Pranav)

Turns the local Decision Tree script (`DecisionTreeModel.py`, Renske) into a
Vertex AI pipeline that trains several models, picks the best one and saves it
to Cloud Storage for the prediction API.

## Steps

```
load_data → split_data → train_model ×4 (depth 2, 3, 4, None, in parallel)
                       → evaluate_model ×4 → select_best_model → save_model
```

| Step | Input | Output |
|---|---|---|
| `load_data` | `gs://group7-iris-data/iris.csv` | checked dataset |
| `split_data` | dataset | train set (80%), test set (20%), stratified, seed 42 |
| `train_model` | train set, max_depth | trained Decision Tree |
| `evaluate_model` | model, test set | accuracy (shown in Vertex UI) |
| `select_best_model` | 4 models + 4 accuracies | best model (ties → simpler tree) |
| `save_model` | best model | `gs://group7-iris-models/model/model.joblib` + a timestamped copy in `model/versions/` |

## Agreements with the rest of the team

- **API (Himey):** the model is always at `gs://group7-iris-models/model/model.joblib`
  (joblib, scikit-learn 1.4.1.post1, numeric labels 0 = setosa, 1 = versicolor, 2 = virginica).
- **Data:** `data/iris.csv` (150 rows, numeric `species` column) must be uploaded
  to `gs://group7-iris-data/iris.csv`.
- **Cloud Build (Charlotte):** compile with `compile_pipeline.py`, submit with `run_pipeline.py`.

## Files

| File | Purpose |
|---|---|
| `iris_pipeline.py` | the pipeline: components + how they connect |
| `compile_pipeline.py` | compiles the pipeline into `iris_decision_tree_pipeline.yaml` |
| `iris_decision_tree_pipeline.yaml` | compiled pipeline, the file Vertex AI runs (committed for versioning) |
| `run_pipeline.py` | submits the YAML to Vertex AI |
| `requirements.txt` | packages for compiling/submitting (kfp, google-cloud-aiplatform) |

## Run by hand

From the repository root:

```bash
pip install -r pipeline/requirements.txt
gsutil cp data/iris.csv gs://group7-iris-data/iris.csv   # once
python pipeline/compile_pipeline.py
python pipeline/run_pipeline.py --project <PROJECT_ID>
```

Then watch the run under **Vertex AI → Pipelines** in the Google Cloud console.
Caching is off by default so every run really retrains; add `--enable-caching`
to reuse unchanged steps, or `--wait` to wait for the run to finish.

## Suggested `cloudbuild/cloudbuild-train.yaml` step

```yaml
substitutions:
  _REGION: europe-west4
  _PIPELINE_ROOT: gs://group7-iris-models/pipeline_root

steps:
  - name: 'python:3.10-slim'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r pipeline/requirements.txt
        python pipeline/compile_pipeline.py
        python pipeline/run_pipeline.py --project $PROJECT_ID --region ${_REGION} --pipeline-root ${_PIPELINE_ROOT}

timeout: '1200s'
```

The Cloud Build service account needs **Vertex AI User**, **Storage Object Admin**
and **Service Account User**. The account the pipeline runs as (by default the
Compute Engine default service account) needs read access to the data bucket and
write access to the model bucket.
