import base64
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer  # or your tokenizer

from models.src_informative.model import InformativeNet
from .predict import predict
from .utils.attention import extract_attention_weights
from .utils.gradcam import GradCAM

# --- Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Informative Model
informative_model = InformativeNet(num_classes=2)
informative_model.load_state_dict(torch.load("models/src_informative/best_informative_model.pth", map_location=device))
informative_model.to(device)
informative_model.eval()

# --- Tokenizer (adjust to your model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# --- Image Transform
image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def predict_explain(text: str, image_bytes: bytes):
    # --- Base prediction
    prediction = predict(text, image_bytes)

    image_explanation = None
    text_explanation = None

    if prediction.get("informative") == "informative":
        # Load and preprocess image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        # Tokenize text and move tensors to device
        encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Grad-CAM for image
        grad_cam = GradCAM(informative_model.resnet.layer4[-1])
        grad_cam.model = informative_model
        # Pass image tensor only; if your GradCAM supports text + image, modify accordingly
        heatmap = grad_cam(input_ids, attention_mask, image_tensor)

        overlay = grad_cam.overlay(image, heatmap)

        # Encode overlay image as base64
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        image_explanation = base64.b64encode(buffer.getvalue()).decode()

        # Extract attention weights from BERT (pass tokenized tensors)
        text_explanation = extract_attention_weights(informative_model.bert, input_ids, attention_mask)

    return {**prediction, "image_explanation": image_explanation, "text_explanation": text_explanation, }
