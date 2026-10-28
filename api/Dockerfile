# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API application code and local model if available
COPY app.py .
# Optional: if model.pkl is in the root or api folder, copy it too
#COPY model.pkl .

# Expose the port Cloud Run uses
EXPOSE 8080

# Run FastAPI using Uvicorn on port 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
