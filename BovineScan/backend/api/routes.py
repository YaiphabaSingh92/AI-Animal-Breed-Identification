from fastapi import APIRouter, File, UploadFile
from backend.core.inference import get_classifier
from PIL import Image
import io

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        classifier = get_classifier()
        result = classifier.predict(image)
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
