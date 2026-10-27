
---

#
# ☁️ Cloud Resources

| Resource | Value |
|---|---|
| Google Cloud Project ID | DE2026 |
| Region | europe-west4 |
| Data Bucket | gs://group7-iris-data |
| Model Bucket | gs://group7-iris-models |
| Artifact Registry Repo | mlops-repo-group7 |
| API Service Name | iris-api |
| UI Service Name | iris-ui |

---

#
# 🔗 Service URLs

| Service | URL |
|---|---|
| Prediction API | https://iris-api-446386690103.europe-west4.run.app |
| Prediction UI | https://iris-ui-xxxxx-ew.a.run.app |

---

#
# 🚀 CI/CD Pipelines

| File | Purpose |
|---|---|
| `cloudbuild/cloudbuild-api.yaml` | Builds and deploys the prediction API to Cloud Run |
| `cloudbuild/cloudbuild-ui.yaml` | Builds and deploys the prediction UI to Cloud Run |
| `cloudbuild/cloudbuild-train.yaml` | Compiles and runs the Vertex AI training pipeline |

#
## Trigger Conditions

| Trigger | Fires When |
|---|---|
| API Deploy Trigger | Code pushed to `api/` folder on main branch |
| UI Deploy Trigger | Code pushed to `ui/` folder on main branch |
| ML Training Trigger | Code pushed to `pipeline/` folder or run manually |

---

#
# 🤖 ML Pipeline

- **Dataset:** Iris flower dataset (loaded from `sklearn.datasets`)
- **Model:** Decision Tree Classifier
- **Features:** sepal_length, sepal_width, petal_length, petal_width
- **Target:** Iris species (setosa, versicolor, virginica)
- **Pipeline:** Vertex AI pipeline compiles, trains, evaluates, 
  and saves best model to Cloud Storage

---

#
# 🔮 API Usage

#
## Health Check
