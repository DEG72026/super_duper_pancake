# Use the same Python version as local development.
FROM python:3.12-slim

# Show logs immediately and avoid generating Python cache files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first so Docker can reuse this build step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed by the UI.
COPY app.py .
COPY templates/ templates/

# Run the application as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

# The port used by the web server inside the container, see ENV.
EXPOSE 8080

# Cloud Run supplies PORT. Gunicorn loads the Flask app from app.py.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 2 --timeout 60 app:app #guess that 2 threads are enough but can raise it