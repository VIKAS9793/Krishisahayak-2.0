# KrishiSahayak API Guide

> **Note**: This document is part of the KrishiSahayak documentation. For general setup and usage, see the [main documentation](README.md).

This guide describes the FastAPI-based REST API for the KrishiSahayak system, which provides programmatic access to the plant disease classification capabilities.

## 📋 Features

- **RESTful API** for plant disease classification
- **Interactive Documentation** via Swagger UI (`/docs`) and ReDoc (`/redoc`)
- **Health Checks** at `/health` endpoint
- **Asynchronous** request handling with FastAPI
- **Batch Processing** support for multiple images
- **Configurable Model Path** via environment variables
- **CORS** enabled for web applications

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- CUDA-compatible GPU (recommended for inference)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   ```

3. Install the package with API dependencies:
   ```bash
   pip install -e ".[api]"
   ```

4. Create a `.env` file in the project root:
   ```env
   # Required
   DEFAULT_MODEL_PATH="models/checkpoints/best.ckpt"
   
   # Optional
   API_HOST="0.0.0.0"
   API_PORT=8000
   API_WORKERS=1
   API_RELOAD=true
   LOG_LEVEL="info"
   MODEL_DEVICE="cuda"  # or "cpu"
   ```

## 🏃 Running the API

### Development Mode

Start the API server with auto-reload enabled:

```bash
uvicorn src.krishi_sahayak.api.main:app --reload
```

Access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Production Mode

For production use with multiple workers:

```bash
gunicorn src.krishi_sahayak.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📡 API Endpoints

### `POST /predict`

Classify a plant disease from an image.

**Request**:
- `Content-Type: multipart/form-data`
- `file`: Image file to classify (required, must be an image)

**Response**:
```json
{
  "filename": "uploaded_image.jpg",
  "predictions": [
    {
      "class_name": "Tomato___Late_blight",
      "confidence": 0.987
    },
    {
      "class_name": "Tomato___healthy",
      "confidence": 0.01
    },
    {
      "class_name": "Tomato___Early_blight",
      "confidence": 0.003
    }
  ],
  "model_checkpoint": "path/to/checkpoint.ckpt"
}
```

### `GET /health`

Check API health status and model availability.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEFAULT_MODEL_PATH` | Yes | - | Path to the trained model checkpoint |
| `API_HOST` | No | "0.0.0.0" | Host to bind the server to |
| `API_PORT` | No | 8000 | Port to run the server on |
| `API_WORKERS` | No | 1 | Number of worker processes |
| `API_RELOAD` | No | true | Enable auto-reload in development |
| `LOG_LEVEL` | No | "info" | Logging level |
| `DEFAULT_MODEL_PATH` | Yes | - | Path to the trained model checkpoint |
| `API_HOST` | No | "0.0.0.0" | Host to bind the server to |
| `API_PORT` | No | 8000 | Port to run the server on |
| `LOG_LEVEL` | No | "info" | Logging level (debug, info, warning, error) |

## 🧪 Testing

Run the API tests:

```bash
pytest tests/api/
```

## 🚀 Deployment

### Docker

Build and run the API using Docker:

```bash
docker build -t krishi-sahayak-api -f Dockerfile.api .
docker run -p 8000:8000 --gpus all krishi-sahayak-api
```

### Kubernetes

Example deployment configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: krishi-sahayak-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: krishi-sahayak-api
  template:
    metadata:
      labels:
        app: krishi-sahayak-api
    spec:
      containers:
      - name: api
        image: krishi-sahayak-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEFAULT_MODEL_PATH
          value: "/app/models/checkpoints/best.ckpt"
        resources:
          limits:
            nvidia.com/gpu: 1
```

```bash
uvicorn src.krishi_sahayak.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

For production deployment using Gunicorn with Uvicorn workers:

```bash
gunicorn -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --log-level info \
  src.krishi_sahayak.api.main:app
```

### Using Docker

Build and run the API using Docker:

```bash
# Build the image
docker build -t krishi-sahayak-api -f Dockerfile.api .

# Run the container
docker run -p 8000:8000 --gpus all krishi-sahayak-api
```

Or using Docker Compose:

```bash
docker-compose -f docker-compose.dev.yaml up --build
```

## 📚 API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication

Currently, the API is open and doesn't require authentication. For production deployments, consider adding API key authentication.

### Endpoints

#### Health Check

```http
GET /health
```

**Response**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model": "MobileNetV3-Large",
  "device": "cuda:0"
}
```

#### Predict from Image

```http
POST /predict
Content-Type: multipart/form-data
```

**Request Body**
- `file`: (required) Image file (JPEG, PNG, WEBP)

**Response**
```json
{
  "filename": "test.jpg",
  "predictions": [
    {
      "class_id": 0,
      "class_name": "Apple___Apple_scab",
      "confidence": 0.98,
      "bbox": [10, 20, 100, 200],
      "confidence_threshold": 0.7
    }
  ],
  "model_checkpoint": "best.ckpt",
  "inference_time_ms": 120.5
}
```

## 🔒 Rate Limiting

The API includes basic rate limiting (100 requests/minute) to prevent abuse. This can be configured via environment variables.

## 📊 Monitoring

### Logs
Logs are written to `logs/api.log` in JSON format and include:
- Request/response details
- Performance metrics
- Error stack traces

### Metrics
Prometheus metrics are available at `/metrics` for monitoring:
- Request count and duration
- Error rates
- Resource usage

## 🤝 Contributing

See the [main contribution guide](../CONTRIBUTING.md) for details on how to contribute to the API.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.