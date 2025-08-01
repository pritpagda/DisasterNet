import base64
from io import BytesIO

import torch
from PIL import Image

from .pred import (
    predict,
    informative_model,
    humanitarian_model,
    damage_model,
    tokenizer,
    image_transform,
)
from .utils.attention import extract_attention_weights
from .utils.gradcam import GradCAM


def predict_explain(text: str, image_bytes: bytes, user_id: int = None, db=None, image_filename: str = None):
    prediction_result = predict(
        text=text,
        image_bytes=image_bytes,
        user_id=user_id,
        db=db,
        image_filename=image_filename,
    )

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(informative_model.device)
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"].to(informative_model.device)
    attention_mask = encoding["attention_mask"].to(informative_model.device)

    image_explanations = {}
    text_explanations = {}

    grad_cam_informative = GradCAM(informative_model, informative_model.resnet.layer4[-1])
    heatmap = grad_cam_informative(input_ids, attention_mask, image_tensor)
    overlay = grad_cam_informative.overlay(image, heatmap)
    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    image_explanations["informative"] = base64.b64encode(buffer.getvalue()).decode()
    text_explanations["informative"] = extract_attention_weights(
        informative_model.bert, input_ids, attention_mask, tokenizer
    )

    if prediction_result.get("humanitarian"):
        grad_cam_humanitarian = GradCAM(humanitarian_model, humanitarian_model.resnet.layer4[-1])
        heatmap = grad_cam_humanitarian(input_ids, attention_mask, image_tensor)
        overlay = grad_cam_humanitarian.overlay(image, heatmap)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        image_explanations["humanitarian"] = base64.b64encode(buffer.getvalue()).decode()
        text_explanations["humanitarian"] = extract_attention_weights(
            humanitarian_model.bert, input_ids, attention_mask, tokenizer
        )

    if prediction_result.get("damage"):
        grad_cam_damage = GradCAM(damage_model, damage_model.resnet.layer4[-1])
        heatmap = grad_cam_damage(input_ids, attention_mask, image_tensor)
        overlay = grad_cam_damage.overlay(image, heatmap)
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        image_explanations["damage"] = base64.b64encode(buffer.getvalue()).decode()
        text_explanations["damage"] = extract_attention_weights(
            damage_model.bert, input_ids, attention_mask, tokenizer
        )

    return {
        **prediction_result,
        "image_explanations": image_explanations,
        "text_explanations": text_explanations,
    }