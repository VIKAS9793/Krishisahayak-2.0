# src/krishi_sahayak/api/main.py
"""Main FastAPI application for the KrishiSahayak API."""
import base64
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Callable

import structlog
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile

from krishi_sahayak.inference.handler import InferenceHandler
from krishi_sahayak.utils.hardware import auto_detect_accelerator
from .config import settings
from .schemas import HealthCheckResponse, Prediction, PredictionResponse

# --- Structured Logging Setup ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

# --- Application Setup ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An AI-Powered Crop Health Assistant API. Provides predictions for plant diseases.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: Dict[str, InferenceHandler] = {}

# --- Middleware for Request Context & Logging ---
@app.middleware("http")
async def add_request_context(request: Request, call_next: Callable) -> Response:
    """Injects a request_id and other context into all logs for this request."""
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        client_ip=request.client.host,
        http_method=request.method,
        http_path=request.url.path,
    )
    start_time = time.monotonic()
    
    response = await call_next(request)
    
    duration_ms = (time.monotonic() - start_time) * 1000
    log.info(
        "request_finished",
        http_status=response.status_code,
        duration_ms=round(duration_ms, 2)
    )
    return response

# --- Utility Functions ---
def _strip_exif_data(image_path: Path) -> None:
    """Strips EXIF metadata from an image in-place to protect user privacy."""
    try:
        image = Image.open(image_path)
        image_data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(image_data)
        image_without_exif.save(image_path)
        log.info("exif_data_stripped", image_path=str(image_path))
    except Exception as e:
        log.warning("exif_strip_failed", image_path=str(image_path), error=str(e))

# --- Dependencies ---
def get_inference_handler() -> InferenceHandler:
    """Dependency injection to load and reuse the model handler."""
    if "handler" not in _model_cache:
        log.info("model_cache_miss", message="Initializing InferenceHandler and loading model...")
        device = torch.device(auto_detect_accelerator())
        model_path = Path(settings.DEFAULT_MODEL_PATH)
        if not model_path.exists():
            log.critical("model_checkpoint_not_found", path=str(model_path))
            raise RuntimeError(f"Model checkpoint not found at: {model_path}")
        _model_cache["handler"] = InferenceHandler(model_path, device)
        log.info("model_loaded_successfully", device=str(device))
    return _model_cache["handler"]

# --- Events ---
@app.on_event("startup")
def startup_event():
    """Preloads the model on application startup to avoid cold starts."""
    get_inference_handler()
    log.info("application_startup_complete", message="API is ready to serve requests.")

# --- Endpoints ---
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """Performs a health check of the API and model availability."""
    handler = get_inference_handler()
    return HealthCheckResponse(status="healthy", version=settings.VERSION, device=str(handler.device))

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    file: UploadFile = File(..., description="Image file of the plant leaf."),
    handler: InferenceHandler = Depends(get_inference_handler)
):
    """Runs inference on an uploaded image and returns disease predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    tmp_path = None
    log.info("prediction_request_received", filename=file.filename, content_type=file.content_type)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or ".jpg").suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        _strip_exif_data(tmp_path)
        
        result = handler.run_single(image_path=tmp_path, nir_image_path=None, top_k=3)
        
        api_predictions = [
            Prediction(
                class_name=p["class"],
                display_name=p["class"].replace("___", " ").replace("__", " "),
                confidence=p["confidence"]
            ) for p in result["predictions"]
        ]
        
        explanation_b64 = None
        if result.get("explanation_image") is not None:
            with tempfile.SpooledTemporaryFile() as buffered:
                result["explanation_image"].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                explanation_b64 = f"data:image/png;base64,{img_str}"

        log.info("prediction_successful", filename=file.filename, top_prediction=api_predictions[0].class_name)
        return PredictionResponse(
            filename=file.filename or "unknown",
            predictions=api_predictions,
            model_checkpoint=str(Path(result["model_checkpoint"]).name),
            explanation_image=explanation_b64
        )
    except Exception as e:
        log.error("prediction_failed", filename=file.filename, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
            log.info("temp_file_deleted", path=str(tmp_path))