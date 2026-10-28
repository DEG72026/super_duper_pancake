# Iris prediction UI

## Current status

The Flask UI accepts four flower measurements in centimetres.
It validates the input, displays the accepted measurements and sends valid
measurements to the prediction API. The predicted species is displayed when
the API returns a valid response.

The API connection has been tested locally and through the deployed UI.
The deployed API reports that the model is loaded. Three reference examples,
one per Iris class, returned the expected species.

## Input validation

All four measurements are required and must be numeric and finite.

Allowed ranges use the full scikit-learn Iris dataset:
observed minimum × 0.8 and observed maximum × 1.2.

| Feature | Minimum (cm) | Maximum (cm) |
|---|---:|---:|
| sepal_length | 3.44 | 9.48 |
| sepal_width | 1.60 | 5.28 |
| petal_length | 0.80 | 8.28 |
| petal_width | 0.08 | 3.00 |

The browser and Python server use the same limits.
These checks catch some unit errors, but cannot detect every incorrect input.
Submitted values remain visible after submission.

## API configuration and behaviour

Set `PREDICTION_API_URL` to the prediction API's base URL, for example
`http://127.0.0.1:8000`. The UI appends `/predict`; a URL already ending in
`/predict` is also accepted. Set the variable before starting the UI and restart
the application after changing it.

The UI sends the four feature names from the table above as a JSON object
containing numeric values. It expects `class_id` and `class_name` in the response,
with the mapping 0 = setosa, 1 = versicolor, 2 = virginica.

Each API request uses a 10-second timeout. The UI displays a clear error if the
API URL is missing, the request times out, the API is unreachable, the API returns
an HTTP error, or its response is invalid. Submitted measurements remain visible.

The `/health` endpoint reports whether the UI server can respond. It does not
check the prediction API or model availability.

## Run locally

Run from the repository root with the project virtual environment active:

```powershell
python -m pip install -r ui/requirements.txt
$env:PREDICTION_API_URL = "http://127.0.0.1:8000"
python ui/app.py
```

Gunicorn is used inside the Linux container, not directly on Windows.

Open http://127.0.0.1:5001.

## Build and run with Docker

Before building the API locally, ensure api/model.pkl contains the trained model.
Cloud Build downloads this file automatically; a local Docker build does not.


Run from the repository root:

```powershell
docker build -t iris-ui:local ./ui
docker run -d --rm --name iris-ui-test -p 127.0.0.1:5002:8080 -e PREDICTION_API_URL=http://host.docker.internal:8000 iris-ui:local
```

Open http://127.0.0.1:5002.

This Docker Desktop example assumes the API is reachable on port 8000 on the
Windows host. `host.docker.internal` lets the UI container reach that host.
For a deployed API, pass its base URL instead. In Cloud Run, configure
`PREDICTION_API_URL` as a service environment variable.

To stop the container, run in another terminal:

```powershell
docker stop iris-ui-test
```

## Start the API locally for integration testing

Run from the repository root with Docker Desktop running:

```powershell
docker build -t iris-api:local ./api
docker run -d --rm --name iris-api-test -p 127.0.0.1:8000:8080 iris-api:local
```

Start the UI using either method above and submit valid measurements. To test
connection failure handling, stop the API and submit again:

```powershell
docker stop iris-api-test
```

The UI should display an API error and preserve the submitted measurements.
Restart the API using the same `docker run` command when needed.

Initial local integration checks used the API's fixed setosa fallback response.
These checked communication and display only. Subsequent deployed checks used
the loaded model and returned the expected species for three reference examples.

## Run doctests

Locally, with the project virtual environment active:

```powershell
python -m doctest -v ui/app.py
```

Or inside the built image:

```powershell
docker run --rm iris-ui:local python -m doctest -v app.py
```

The doctest for a valid form submission uses a simulated API response, so the
API does not need to be running for doctests. These tests alone do not verify
live HTTP communication.

## Verification completed

- Local form submission checked.
- Docker image built successfully.
- Form submission through the Docker container checked.
- All 21 doctests passed inside the UI container.
- Local UI-to-API communication checked, including with the UI in Docker.
- Stopping the local API produced the expected error and preserved the input.
- Dependencies pinned to the versions from the tested image.

## Deployed application verification

- The deployed API health endpoint reports `model_loaded: true`.
- The deployed UI successfully communicates with the prediction API.
- Three reference examples, one per Iris class, returned the expected species:
  setosa, versicolor and virginica.
- These checks verify the deployed prediction flow; model accuracy is
  evaluated separately in the training pipeline.
