import base64
from io import BytesIO
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from colpali_engine.models.paligemma.colpali import ColPaliForConditionalGeneration
from colpali_engine.utils.processing_utils import get_paligemma_preprocessor

# Placeholder for model loading
# In a real application, you would load your trained model here
# For example:
# model_name = "google/paligemma-3b-pt-224"
# model = ColPaliForConditionalGeneration.from_pretrained(model_name)
# processor = get_paligemma_preprocessor(model_name)

# For now, let's use a mock model and processor
class MockModel:
    def to(self, device):
        return self

    def encode_image(self, pixel_values, **kwargs):
        return torch.randn(len(pixel_values), 1024)

    def encode_text(self, input_ids, attention_mask, **kwargs):
        return torch.randn(len(input_ids), 1024)

class MockProcessor:
    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        if text:
            return {"input_ids": torch.randint(0, 1000, (len(text), 10)), "attention_mask": torch.ones(len(text), 10)}
        if images:
            return {"pixel_values": torch.randn(len(images), 3, 224, 224)}


model = MockModel()
processor = MockProcessor()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI()

class TextRequest(BaseModel):
    queries: List[str]

class ImageRequest(BaseModel):
    images: List[str]  # Base64 encoded images

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/embed/text", response_model=EmbeddingsResponse)
async def embed_text(request: TextRequest):
    try:
        inputs = processor(text=request.queries, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.encode_text(**inputs).cpu().tolist()
        return EmbeddingsResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/image", response_model=EmbeddingsResponse)
async def embed_image(request: ImageRequest):
    images = []
    for img_b64 in request.images:
        try:
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    try:
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.encode_image(**inputs).cpu().tolist()
        return EmbeddingsResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "ColPali embedding server is running"}
