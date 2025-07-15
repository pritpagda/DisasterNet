import base64
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.src_informative.model import InformativeNet
from models.src_humanitarian.model import HumanitarianNetV1
from models.src_damage.model import DamageNetV1

from .predict import predict
from .utils.attention import extract_attention_weights
from .utils.gradcam import GradCAM

# --- Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models
informative_model = InformativeNet(num_classes=2)
informative_model.load_state_dict(torch.load("models/src_informative/best_informative_model.pth", map_location=device))
informative_model.to(device)
informative_model.eval()

humanitarian_model = HumanitarianNetV1(num_classes=7)
humanitarian_model.load_state_dict(torch.load("models/src_humanitarian/best_humanitarian_model.pth", map_location=device))
humanitarian_model.to(device)
humanitarian_model.eval()

damage_model = DamageNetV1(num_classes=3)
damage_model.load_state_dict(torch.load("models/src_damage/best_damage_model.pth", map_location=device))
damage_model.to(device)
damage_model.eval()

# --- Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# --- Image Transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_explain(text: str, image_bytes: bytes):
    # --- Prediction step
    prediction = predict(text, image_bytes)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # --- Tokenization
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    image_explanations = {}
    text_explanations = {}

    # --- Informative explanation
    grad_cam_informative = GradCAM(informative_model, informative_model.resnet.layer4[-1])
    heatmap = grad_cam_informative(input_ids, attention_mask, image_tensor)
    overlay = grad_cam_informative.overlay(image, heatmap)
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    image_explanations["informative"] = base64.b64encode(buffer.getvalue()).decode()
    text_explanations["informative"] = extract_attention_weights(informative_model.bert, input_ids, attention_mask, tokenizer)

    # --- Humanitarian explanation
    if prediction.get("humanitarian"):
        grad_cam_humanitarian = GradCAM(humanitarian_model, humanitarian_model.resnet.layer4[-1])
        heatmap = grad_cam_humanitarian(input_ids, attention_mask, image_tensor)
        overlay = grad_cam_humanitarian.overlay(image, heatmap)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        image_explanations["humanitarian"] = base64.b64encode(buffer.getvalue()).decode()
        text_explanations["humanitarian"] = extract_attention_weights(humanitarian_model.bert, input_ids, attention_mask, tokenizer)

    # --- Damage explanation
    if prediction.get("damage"):
        grad_cam_damage = GradCAM(damage_model, damage_model.resnet.layer4[-1])
        heatmap = grad_cam_damage(input_ids, attention_mask, image_tensor)
        overlay = grad_cam_damage.overlay(image, heatmap)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        image_explanations["damage"] = base64.b64encode(buffer.getvalue()).decode()
        text_explanations["damage"] = extract_attention_weights(damage_model.bert, input_ids, attention_mask, tokenizer)

    return {
        **prediction,
        "image_explanations": image_explanations,
        "text_explanations": text_explanations
    }
