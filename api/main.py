import base64
from io import BytesIO
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

# Load the model and processor
model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(model_name)
device = model.device

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class QueryRequest(BaseModel):
    query: str

class ImagesRequest(BaseModel):
    images: List[str]  # Base64 encoded images

class QueriesRequest(BaseModel):
    queries: List[str]

class ScoreRequest(BaseModel):
    image_embeddings: List[List[List[float]]]
    query_embeddings: List[List[List[float]]]

class EmbeddingResponse(BaseModel):
    embedding: List[List[float]]

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[List[float]]]

class ScoresResponse(BaseModel):
    scores: List[List[float]]

def decode_image(img_b64: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(img_b64)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

@app.post("/process/image", response_model=EmbeddingResponse)
async def process_image(request: ImageRequest):
    image = decode_image(request.image)
    batch_images = processor.process_images([image]).to(device)
    with torch.no_grad():
        image_embedding = model(**batch_images).cpu().tolist()
    return EmbeddingResponse(embedding=image_embedding)

@app.post("/process/query", response_model=EmbeddingResponse)
async def process_query(request: QueryRequest):
    batch_queries = processor.process_queries([request.query]).to(device)
    with torch.no_grad():
        query_embedding = model(**batch_queries).cpu().tolist()
    return EmbeddingResponse(embedding=query_embedding)

@app.post("/process/images", response_model=EmbeddingsResponse)
async def process_images(request: ImagesRequest):
    images = [decode_image(img_b64) for img_b64 in request.images]
    batch_images = processor.process_images(images).to(device)
    with torch.no_grad():
        image_embeddings = model(**batch_images).cpu().tolist()
    return EmbeddingsResponse(embeddings=image_embeddings)

@app.post("/process/queries", response_model=EmbeddingsResponse)
async def process_queries(request: QueriesRequest):
    batch_queries = processor.process_queries(request.queries).to(device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries).cpu().tolist()
    return EmbeddingsResponse(embeddings=query_embeddings)

@app.post("/process/score_multi_vector", response_model=ScoresResponse)
async def score_multi_vector(request: ScoreRequest):
    try:
        image_embeddings = torch.tensor(request.image_embeddings, device=device)
        query_embeddings = torch.tensor(request.query_embeddings, device=device)
        scores = processor.score_multi_vector(query_embeddings, image_embeddings).cpu().tolist()
        return ScoresResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "ColPali embedding server is running"}
