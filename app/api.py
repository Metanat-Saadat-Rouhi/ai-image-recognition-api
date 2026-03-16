from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import ImageClassifier
from app.utils import load_image_from_bytes

app = FastAPI(
    title="AI Image Recognition API",
    description="Upload an image and get a prediction using a pretrained ResNet18 model.",
    version="1.0.0",
)

classifier = ImageClassifier()


@app.get("/")
def home():
    return {
        "message": "AI Image Recognition API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = load_image_from_bytes(image_bytes)
        image_tensor = classifier.preprocess(image).unsqueeze(0)
        result = classifier.predict(image_tensor)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "top_5": result["top_5"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")