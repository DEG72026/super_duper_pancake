# Iris prediction UI

## Current status

The Flask UI accepts four flower measurements in centimetres.
It validates the input and displays the accepted measurements.

The prediction API is not connected yet. No model predictions are made.

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

## Run locally

Run from the repository root with the project virtual environment active:

```powershell
python -m pip install -r ui/requirements.txt
python ui/app.py
```

Gunicorn is used inside the Linux container, not directly on Windows.

Open http://127.0.0.1:5001.

## Build and run with Docker

Run from the repository root:

```powershell
docker build -t iris-ui:local ./ui
docker run --rm --name iris-ui-test -p 127.0.0.1:5002:8080 iris-ui:local
```

Open http://127.0.0.1:5002.

To stop the container, run in another terminal:

```powershell
docker stop iris-ui-test
```

## Run doctests

Locally, with the project virtual environment active:

```powershell
python -m doctest -v ui/app.py
```

Or inside the built image:

```powershell
docker run --rm iris-ui:local python -m doctest -v app.py
```

## Verification completed

- Local form submission checked.
- Docker image built successfully.
- Form submission through the Docker container checked.
- All doctests passed inside the container.
- Dependencies pinned to the versions from the tested image.

## Remaining integration work

- Agree on the prediction API request and response formats with Himey.
- Add a configurable prediction API URL and request timeout.
- Display predictions and handle API failures.
- Coordinate Cloud Build and Cloud Run deployment with Charlotte.
- Test the complete deployed application.