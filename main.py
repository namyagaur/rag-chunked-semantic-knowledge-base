from pathlib import Path

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
            }
        })
    return chunks
for file in files:
    chunk_list.extend(chunk_text(file,200,50))

