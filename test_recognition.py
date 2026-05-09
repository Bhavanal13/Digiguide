import os
import faiss
import json
import numpy as np
import argparse
from digiguide.utils import image_to_embedding

# Paths (assumed same directory structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "vector_store.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")

def load_index_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Index or metadata file not found. Run build_index.py first.")
        
    index = faiss.read_index(INDEX_FILE)
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
        # JSON keys are always strings, convert to int for ID lookups
        metadata = {int(k): v for k, v in metadata.items()}
        
    return index, metadata

def recognize_image(image_path, k=1):
    index, metadata = load_index_and_metadata()
    
    # Generate embedding for query image
    query_emb = image_to_embedding(image_path)
    # query_emb is (1, 512) and type float32
    query_emb = query_emb.astype('float32')
    
    # Search
    distances, indices = index.search(query_emb, k)
    
    print(f"Results for '{image_path}':")
    for i in range(k):
        idx = indices[0][i]
        score = distances[0][i]
        
        if idx in metadata:
            info = metadata[idx]
            print(f"Rank {i+1}: {info['landmark_name']} (Score: {score:.4f}) - {info['filename']}")
        else:
            print(f"Rank {i+1}: Unknown Index {idx} (Score: {score:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test landmark recognition")
    parser.add_argument("--image", type=str, required=True, help="Path to image to test")
    args = parser.parse_args()
    
    recognize_image(args.image)
