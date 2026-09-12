from pathlib import Path
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding_model(text):
    return model.encode(text)

files = Path("documents").glob("*.txt")

chunk_list = []

def chunk_text(source,chunk_size,overlap):
    chunks = []
    text = source.read_text()
    for chunkid,i in enumerate(range(0,len(text),chunk_size-overlap)):
        chunks.append({
            "content": text[i:i+chunk_size],
            "metadata":{
                "source" : source,
                "chunkid": chunkid
            },
            "embedding": embedding_model(text[i:i+chunk_size])
        })
    return chunks
for file in files:
    chunk_list.extend(chunk_text(file,200,50))

