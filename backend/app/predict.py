import sys
import os
import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import base64
import io

# Add models/src_informative to Python path (adjust relative to this file location)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/src_informative")))

from model import DisasterNetV1  # Your model class

# Initialize tokenizer and model once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = DisasterNetV1()

# Load model weights
MODEL_WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/src_informative/best_model.pth"))
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

# Define image transforms as expected by ResNet50
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

def preprocess_image(image_base64: str):
    """Decode base64 image and apply transforms"""
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image_transform(image).unsqueeze(0)  # add batch dimension

def predict(text: str, image_base64: str):
    """Run prediction given text and base64-encoded image"""
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Preprocess image
    image_tensor = preprocess_image(image_base64)

    # Run inference
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image=image_tensor
        )
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "disaster" if prediction.item() == 1 else "not_disaster"

    return {
        "label": label,
        "confidence": round(confidence.item(), 3)
    }
