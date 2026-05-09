import os
import faiss
import numpy as np
import json
from digiguide.utils import image_to_embedding

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
INDEX_FILE = os.path.join(BASE_DIR, "vector_store.index")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")
LOCATIONS_FILE = os.path.join(BASE_DIR, "locations.json")

def load_locations():
    if os.path.exists(LOCATIONS_FILE):
        with open(LOCATIONS_FILE, 'r') as f:
            return json.load(f)
    print("Warning: locations.json not found.")
    return {}

def build_index():
    print("Building index derived from images in:", DATASET_DIR)
    
    locations = load_locations()
    
    # Store metadata: index_id -> info
    metadata = {}
    
    # List to hold all embeddings
    embeddings = []
    
    current_id = 0
    
    # Walk through the dataset directory
    # Structure: dataset/landmark_name/image.jpg
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory {DATASET_DIR} not found!")
        return

    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for landmark_name in classes:
        class_dir = os.path.join(DATASET_DIR, landmark_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {landmark_name}: {len(images)} images")
        
        # Get location data for this landmark
        loc_data = locations.get(landmark_name, {})
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            
            # Generate embedding
            emb = image_to_embedding(img_path)
            
            # emb is shape (1, 512), we need to flatten for list or keep as (1, 512) for vstack
            embeddings.append(emb)
            
            # specific mapping
            meta_entry = {
                "landmark_name": landmark_name,
                "filename": img_file,
                # Add location data if available
                "lat": loc_data.get("lat"),
                "lon": loc_data.get("lon")
            }
            metadata[current_id] = meta_entry
            
            current_id += 1

    if not embeddings:
        print("No images found to index.")
        return

    # Convert list of arrays to single numpy array
    # each emb is (1, dim)
    embedding_matrix = np.vstack(embeddings).astype('float32')
    
    d = embedding_matrix.shape[1]
    print(f"Embedding dimension: {d}, Count: {len(embeddings)}")
    
    # Create FAISS index
    # Using L2 distance equivalent to inner product on normalized vectors = cosine similarity
    # But usually IndexFlatL2 is fine if we want Euclidean distance.
    # Logic in main.py was L2 normalize -> so dot product (IndexFlatIP) is cosine similarity.
    # Let's use IndexFlatIP for cosine similarity since vectors are normalized.
    index = faiss.IndexFlatIP(d)
    index.add(embedding_matrix)
    
    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved to {INDEX_FILE}")
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {METADATA_FILE}")

if __name__ == "__main__":
    build_index()
