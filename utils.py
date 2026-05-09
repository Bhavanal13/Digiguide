import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None

def get_model_and_processor():
    global _model, _processor
    if _model is None:
        print(f"Loading model {MODEL_NAME} to {device}...")
        try:
            _model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
            _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        except Exception as e:

            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise e
    return _model, _processor


def image_to_embedding(image):
    model, processor = get_model_and_processor()
    
    # helper to process image if it's a path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    inputs = processor(images=image, return_tensors="pt")
    # move inputs to device
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        # L2 normalize
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    
    return emb.cpu().numpy()
