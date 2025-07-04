import os
import pickle
import faiss
import openai
from tqdm import tqdm

# CONFIG
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def get_embedding(text):
    res = openai.Embedding.create(input=[text], model=EMBED_MODEL)
    return res['data'][0]['embedding']

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
                docs.append((filename, content))
    return docs

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    documents = load_documents()
    embeddings = []
    texts = []

    print("Embedding documents...")
    for name, doc in tqdm(documents):
        embedding = get_embedding(doc[:4000])  # truncate to 4K tokens
        embeddings.append(embedding)
        texts.append(doc)

    # FAISS Index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    # Save
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "documents.pkl"), "wb") as f:
        pickle.dump(texts, f)

    print("✅ Embedding & Indexing Complete.")

if __name__ == "__main__":
    import numpy as np
    main()
