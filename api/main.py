"""
FastAPI server for plant disease classification.

Run from project root:
    uvicorn api.main:app --reload
"""
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.inference import PlantClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Initialized in lifespan, reused across all requests
classifier: PlantClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; clean up at shutdown."""
    global classifier
    classifier = PlantClassifier(
        model_path=PROJECT_ROOT / "models" / "mobilenet_v3_small.pth",
        classes_path=PROJECT_ROOT / "models" / "classes.json",
    )
    yield
    classifier = None


app = FastAPI(
    title="Plant Doctor API",
    description="ML-powered plant disease diagnosis from leaf photos.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow browser frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: in production, restrict to actual frontend URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Simple health check."""
    return {"status": "ok", "service": "plant-doctor"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image upload, return top-3 plant disease predictions."""
    # Validate it's actually an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (image/jpeg, image/png, ...)",
        )

    # Read uploaded bytes, parse with PIL
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not open image: {e}",
        )

    # Inference
    predictions = classifier.predict(image, top_k=3)
    return {"predictions": predictions}