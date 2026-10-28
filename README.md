# 🌸 MLOps for an Iris Decision Tree — Group 7

Data Engineering 2026–2027 (JADS), Assignment 1. A Decision Tree that classifies Iris flowers, trained by a Vertex AI pipeline and served through a prediction API and a web UI on Cloud Run, all built and deployed with Cloud Build.

---

# 🔗 Service URLs

| Service | URL |
|---|---|
| Prediction UI | https://iris-ui-446386690103.europe-west4.run.app |
| Prediction API | https://iris-api-446386690103.europe-west4.run.app (`GET /health`, `POST /predict`) |

---

# 📁 Repository Structure

| Folder | Contents |
|---|---|
| `pipeline/` | Vertex AI training pipeline (Kubeflow), pipeline image (Dockerfile), compile / run / local-run scripts — see `pipeline/README.md` |
| `tests/` | Pipeline tests (compilation, model accuracy, quality gate decisions, image configuration) |
| `api/` | FastAPI prediction API and its Dockerfile |
| `test/` | API unit tests |
| `ui/` | Flask UI, Dockerfile and doctests — see `ui/README.md` |
| `cloudbuild/` | Cloud Build files for training, API and UI |
| `data/` | `iris.csv` (150 flowers, 4 features, species 0–2) |

---

# 🤖 ML Pipeline & System Behaviour

1. **Training (Vertex AI):** load and validate data → 80/20 split → four Decision Trees (max_depth 2, 3, 4, None) trained and evaluated in parallel → best model selected → **quality gate** (≥ 90% accuracy, at least as good as the current model, three reference flowers correct) → model saved only if approved.
2. **Model repository:** `gs://group7-iris-models/model/` holds `model.joblib` (current model), `metrics.json` (its accuracy and version) and `versions/` (all released models).
3. **Serving:** the API loads `model.joblib` from the bucket at start-up; the UI sends the four measurements to the API and shows the predicted species.

---

# 🚀 CI/CD Pipelines

| Trigger | Fires on | Runs |
|---|---|---|
| ML-Training-Trigger | changes to `pipeline/`, `tests/`, `data/` or `cloudbuild-train.yaml` | pipeline tests → build & push pipeline image → compile & submit the Vertex AI pipeline |
| ML-Training-Trigger-Scheduled | every Monday at 00:00 (Cloud Scheduler) | same as above |
| API-Deploy-Trigger | changes to `api/` or `test/` | API tests → build → push → deploy `iris-api` |
| API-Deploy-on-New-Model | a new `model/model.joblib` in the bucket (Pub/Sub) | redeploys `iris-api`, which loads the new model |
| UI-Deploy-Trigger | changes to `ui/` | UI doctests → build → push → deploy `iris-ui` |

Every build tests its own component before anything is deployed.

---

# ☁️ Cloud Resources

| Resource | Value |
|---|---|
| Project | `de2026-509009` |
| Region | `europe-west4` |
| Data bucket | `gs://group7-iris-data` |
| Model bucket | `gs://group7-iris-models` |
| Artifact Registry | `mlops-repo-group7` |
| Pub/Sub topic | `iris-model-update` |

---

# 🧪 Running the Tests Locally

```bash
pip install -r tests/requirements.txt && pytest tests/                                   # pipeline
pip install -r api/requirements.txt pytest && python -m pytest test/                      # API
pip install -r ui/requirements.txt pytest && python -m pytest --doctest-modules ui/app.py # UI
```

---
