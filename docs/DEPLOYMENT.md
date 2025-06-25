# Model Deployment Strategy Guide

This guide outlines the proposed strategy for exporting, optimizing, and deploying the KrishiSahayak models for production use. This document describes **future work** and the recommended path forward.

## Table of Contents
1. [Phase 1: Model Export](#1-phase-1-model-export)
2. [Phase 2: Model Optimization](#2-phase-2-model-optimization)
3. [Phase 3: Serving Strategy](#3-phase-3-serving-strategy)
4. [Phase 4: Monitoring](#4-phase-4-monitoring)

## 1. Phase 1: Model Export

The first step is to export the trained PyTorch models from their checkpoint format (`.ckpt`) into a standardized, portable format.

### Recommended Format: ONNX
**ONNX (Open Neural Network Exchange)** is the industry standard. It provides a graph-based representation of the model that is independent of the training framework.

**Proposed Script (`scripts/export_onnx.py`):**
```bash
python scripts/export_onnx.py \
    --checkpoint path/to/student_model.ckpt \
    --output models/student.onnx \
    --input-shape 1 3 224 224 \
    --opset-version 13
```

## 2. Phase 2: Model Optimization

To prepare for deployment on edge devices or for high-throughput serving, the exported ONNX model should be optimized.

### Quantization
Post-training dynamic quantization is recommended as a starting point. It can reduce model size by ~4x and speed up inference with a minimal drop in accuracy.

Example using ONNX Runtime:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8,
)
```

## 3. Phase 3: Serving Strategy

### Proposed API Server: FastAPI
A lightweight FastAPI server is recommended for creating a REST API endpoint.

Example Snippet (`api/main.py`):

```python
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
# ... other imports for preprocessing ...

app = FastAPI()
ort_session = ort.InferenceSession("model_quantized.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Preprocess image from upload
    # 2. Run inference with ONNX Runtime
    # 3. Post-process results
    # 4. Return JSON response
    return {"prediction": "result"}
```

## 4. Phase 4: Monitoring

For a production deployment, monitoring is critical. A system using Prometheus for metrics collection and Grafana for visualization is the standard approach.

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```


### POST /predict

**Request**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response**
```json
{
  "class_id": 12,
  "class_name": "Tomato_Early_Blight",
  "confidence": 0.92,
  "timestamp": "2023-10-25T12:00:00Z"
}
```

## 5. Monitoring

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'krishisahayak'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
```

### Key Metrics
- `inference_latency_seconds`
- `requests_total`
- `predictions_total{class_id="<id>"}`
- `gpu_utilization`
- `gpu_memory_used`

## Versioning

Model versions follow Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality
- PATCH: Backwards-compatible bug fixes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
