import os
import io
import logging
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import mlflow
import mlflow.pyfunc
from torchvision import transforms
import json
import tempfile
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "ivis_resnet")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialize FastAPI app
app = FastAPI(title="Image Classification API", version="1.0.0")

# Global model variable
model = None
transform = None
device = None

def load_model():
    """Load model from MLflow."""
    global model, transform, device
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load model from production stage
        model_uri = f"models:/{MODEL_NAME}/Production"
        logger.info(f"Loading model from {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
	load_class_names()        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ok",
        "model": f"models:/{MODEL_NAME}/Production"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            # MLflow pyfunc models expect numpy arrays
            input_data = image_tensor.numpy()
            predictions = model.predict(input_data)
            
            # Get the predicted class
            if isinstance(predictions, np.ndarray):
                if predictions.ndim > 1:
                    predicted_idx = predictions.argmax(axis=1)[0]
                else:
                    predicted_idx = predictions.argmax()
            else:
                predicted_idx = predictions
            predicted_label = (
            class_names[predicted_idx] if class_names and 0 <= predicted_idx < len(class_names)
            else str(predicted_idx)
            )
        
        # Get confidence scores if available
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            probs = torch.nn.functional.softmax(torch.from_numpy(predictions[0]), dim=0)
            confidence = float(probs[predicted_idx])
        else:
            confidence = 1.0
        
        return JSONResponse(content={
            "prediction": int(predicted_idx),
	    "prediction_label": predicted_label,
            "confidence": confidence,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Image Classification API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
