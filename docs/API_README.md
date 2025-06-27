# KrishiSahayak API Guide

> **Note**: This document provides detailed information about the API. For general project setup, see the [main Quick Start Guide](QUICKSTART.md).

This guide describes the FastAPI-based REST API for the KrishiSahayak system, which provides programmatic access to the plant disease classification capabilities.

## 🚀 Features

- RESTful endpoints for health checks and predictions.
- Interactive documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`).
- Asynchronous request handling for high performance.
- Configuration managed via environment variables (or a `.env` file).
- Production-ready deployment patterns using Gunicorn and Docker.

## 🏃 Running the API

### 1. Installation
Ensure you have installed the project with the `api` dependencies:
```bash
# From the project root directory
pip install -e ".[api]"
```

### 2. Environment Configuration
Create a `.env` file in the project root by copying the template:

```bash
cp .env.example .env
```

Ensure the `DEFAULT_MODEL_PATH` in your `.env` file points to a valid model checkpoint.

### 3. Running the Server
For Development (with auto-reload):

```bash
uvicorn src.krishi_sahayak.api.main:app --reload
```

For Production:
Use a process manager like Gunicorn to manage Uvicorn workers for scalability and resilience.

```bash
gunicorn src.krishi_sahayak.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Access the interactive API documentation at http://localhost:8000/docs.

## 📡 API Endpoints

### Health Check
`GET /health`

Checks the API's status and confirms the model is loaded.

**Success Response (200 OK)**:

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "device": "cuda"
}
```

### Predict Disease
`POST /predict`

Classifies a plant disease from an uploaded image.

**Request**:

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: The image file to be classified (e.g., JPEG, PNG).

**Success Response (200 OK)**:

```json
{
  "filename": "tomato_leaf.jpg",
  "predictions": [
    {
      "class_name": "Tomato___Late_blight",
      "display_name": "Tomato Late blight",
      "confidence": 0.987
    }
  ],
  "model_checkpoint": "best_model.ckpt",
  "explanation_image": "data:image/png;base64,iVBORw0KGgo..."
}
```

## 🐳 Deployment with Docker

The project includes a `Dockerfile.api` optimized for running the service.

### 1. Build the Docker Image:

```bash
docker build -t krishi-sahayak-api -f Dockerfile.api .
```

### 2. Run the Docker Container:

```bash
# For CPU
docker run -d -p 8000:8000 -v /path/to/models:/app/models --env-file .env krishi-sahayak-api

# For NVIDIA GPU
docker run -d -p 8000:8000 --gpus all -v /path/to/models:/app/models --env-file .env krishi-sahayak-api
```

You can also use the provided `docker-compose.yaml` for a streamlined local development experience with live reloading.

## 🔧 Configuration via Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEFAULT_MODEL_PATH` | Yes | Path to the trained model checkpoint (`.ckpt`) |
| `LOG_LEVEL` | No | Logging level (e.g., `INFO`, `DEBUG`) |
| `ENVIRONMENT` | No | Set to `production` or `development` |
