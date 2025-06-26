"""Main FastAPI application for the KrishiSahayak API."""
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

# REFACTORED: Corrected import paths to be relative to the installed package root.
from krishisahayak.inference.handler import InferenceHandler
from krishisahayak.utils.hardware import auto_detect_accelerator
from .config import settings
from .schemas import HealthCheckResponse, PredictionResponse

# --- Application Setup ---
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An AI-Powered Crop Health Assistant API.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# REFACTORED: Added required CORS middleware to allow browser-based access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple in-memory cache for the loaded model handler to prevent reloading.
_model_cache: Dict[str, InferenceHandler] = {}


# --- Dependencies ---

def get_inference_handler() -> InferenceHandler:
    """
    Dependency injection function to load the model on startup and reuse the
    handler instance for all requests.
    """
    # This check ensures the model is loaded only once.
    if "handler" not in _model_cache:
        logger.info("Initializing InferenceHandler and loading model...")
        device = torch.device(auto_detect_accelerator())
        model_path = Path(settings.DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(f"Model checkpoint not found at: {model_path}")
        _model_cache["handler"] = InferenceHandler(model_path, device)
        logger.info(f"Model loaded successfully on device: {device}")
    return _model_cache["handler"]


# --- Events ---

@app.on_event("startup")
def startup_event():
    """Preloads the model on application startup to avoid cold starts."""
    get_inference_handler()
    logger.info("Application startup complete. API is ready to serve requests.")


# --- Endpoints ---

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """Performs a health check of the API and model availability."""
    handler = get_inference_handler()
    return HealthCheckResponse(
        status="healthy",
        version=settings.VERSION,
        device=str(handler.device)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    file: UploadFile = File(..., description="Image file of the plant leaf."),
    handler: InferenceHandler = Depends(get_inference_handler)
):
    """Runs inference on an uploaded image and returns disease predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Use a temporary file to robustly handle the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        logger.info(f"Processing prediction for file: {file.filename}")
        
        # The project's central InferenceHandler contains all logic for prediction.
        result = handler.run_single(
            image_path=tmp_path,
            nir_image_path=None, # Assuming no NIR for this basic API endpoint
            top_k=3
        )
        
        # Reformat the handler's output to match the API's response schema
        api_predictions = [{"class_name": p["class"], "confidence": p["probability"]} for p in result["predictions"]]
        
        return PredictionResponse(
            filename=file.filename,
            predictions=api_predictions,
            model_checkpoint=str(result["model_checkpoint"])
        )

    except Exception as e:
        logger.error(f"Prediction failed for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
    finally:
        # Ensure temporary file is always deleted
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()