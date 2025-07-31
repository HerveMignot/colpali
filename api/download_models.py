import os

# import base64
# import os
# from io import BytesIO
# from typing import List

# import torch
# from fastapi import FastAPI, HTTPException
# from PIL import Image
# from pydantic import BaseModel
# from transformers.utils.import_utils import is_flash_attn_2_available

# from colpali_engine.models import ColQwen2, ColQwen2Processor

#hf_path = "/var/tmp/huggingface"
#os.environ["HF_HOME"] = hf_path
#os.environ["HUGGINGFACE_HUB_CACHE"] = hf_path
#os.environ["TRANSFORMERS_CACHE"] = hf_path
#os.environ["HF_DATASETS_CACHE"] = hf_path

# Download the model and processor

from huggingface_hub import snapshot_download

hf_local_cache_path = os.environ.get('MODEL_CACHE_PATH', '/var/tmp/huggingface')

model_name = "vidore/colqwen2-v1.0"
snapshot_download(repo_id=model_name, cache_dir=hf_local_cache_path)

model_name = "vidore/colqwen2-base"
snapshot_download(repo_id=model_name, cache_dir=hf_local_cache_path)

# # Use this to download files to cache directory
# model = ColQwen2.from_pretrained(
#     model_name,
#     cache_dir=hf_local_cache_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda:0" if torch.cuda.is_available() else "cpu",
#     attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
# ).eval()
# processor = ColQwen2Processor.from_pretrained(
#     model_name,
#     cache_dir=hf_local_cache_path
# )

print('Models downloaded')
