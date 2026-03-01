from endee import Endee, Precision
import numpy as np
import json
from tqdm.auto import tqdm
import os

INDEX_NAME = "gale_medicine_rag"          
EMBEDDINGS_PATH = "/Users/gurlalsingh/Desktop/endee/local/pdf_embeddings.npy"    
METADATA_PATH  = "/Users/gurlalsingh/Desktop/endee/local/pdf_chunks_metadata.json"
BATCH_SIZE = 200

client = Endee()

dimension = None

print("Loading embeddings and metadata")
embeddings = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

assert len(embeddings) == len(metadata_list), "Mismatch between embeddings and metadata!"

dimension = embeddings.shape[1]
print(f"Detected dimension: {dimension}")

try:
    client.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        space_type="cosine",
        precision=Precision.INT8  
    )
    print(f"Index '{INDEX_NAME}' created.")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Index '{INDEX_NAME}' already exists – continuing.")
    else:
        raise e
    
index = client.get_index(INDEX_NAME)

points = []
for i, (vec, meta) in enumerate(zip(embeddings, metadata_list)):
    points.append({
        "id": meta["id"],
        "vector": vec.tolist(),
        "meta": {
            "text": meta["text"],
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "length": meta["length"]
        }
    })

    if len(points) >= BATCH_SIZE or i == len(embeddings) - 1:
        print(f"Upserting batch {len(points)} points (up to {i+1}/{len(embeddings)})...")
        index.upsert(points)
        points = []  

print("Ingestion done!!!")