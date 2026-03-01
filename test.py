from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm   
from endee import Endee, Precision
import json

client = Endee()
embedder = SentenceTransformer("BAAI/bge-m3")

query = "what is Abscess?"
q_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()


INDEX_NAME = "gale_medicine_rag"
index = client.get_index(name=INDEX_NAME)
results = index.query(vector=q_vec, top_k=5)

top_5_list = []

print(f"Query: {query}")
print("Top matches from Endee:")
for rank, res in enumerate(results, 1):
    score = res.get("similarity", "N/A")
    text = res["meta"].get("text", "")
    text_preview = res["meta"]["text"][:400] + "..." if len(res["meta"]["text"]) > 400 else res["meta"]["text"]
    print(f"Rank {rank} | Score: {score:.4f}")
    print(f"Text:\n{text_preview}\n")
    print("-" * 80)

    top_5_list.append({
            "rank": rank,
            "score": float(score) if score != "N/A" else None,
            "similarity": float(score) if score != "N/A" else None,
            "text": text,
            "id": res["meta"].get("id", ""),
            "source": res["meta"].get("source", ""),
            "chunk_index": res["meta"].get("chunk_index", None)
        })

json_filename = "query1.json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump({
        "query": query,
        "model": "BAAI/bge-m3",
        "top_5": top_5_list
    }, f, ensure_ascii=False, indent=2)