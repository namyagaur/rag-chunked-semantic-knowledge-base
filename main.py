from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
def cosine_sim(a,b):
    return (np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))

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

full_doc = chunk_list
def retrieve(query,full_docs,k):
    res = []
    query_emb = embedding_model(query)

    for d in full_docs:
        res.append({
            "source":d["metadata"]["source"],
            "content":d["content"],
            "chunkid": d["metadata"]["chunkid"],
            "score": cosine_sim(query_emb,d["embedding"])
        })

    res.sort(key = lambda x : x["score"], reverse= True)
    return res[:k]


query = input("Enter your query: ")

for rank,chunk in enumerate(retrieve(query,full_doc,5),start=1):
    print(f"----- Rank{rank}-----")
    print("Score:----  ", chunk["source"])
    print("Content:---  ",chunk["content"])
    print("Chunk-ID:---  ",chunk["chunkid"])
    print("Score:---  ",chunk["score"])