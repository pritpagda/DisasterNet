from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from models.src_humanitarian.model import HumanitarianNetV1
from models.src_informative.model import InformativeNet

# --- Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- Image preprocessing
image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

# --- Load Humanitarian Model
humanitarian_model = HumanitarianNetV1(num_classes=7)
humanitarian_model.load_state_dict(
    torch.load("models/src_humanitarian/best_humanitarian_model.pth", map_location=device))
humanitarian_model.to(device)
humanitarian_model.eval()

# --- Load Informative Model
informative_model = InformativeNet(num_classes=2)
informative_model.load_state_dict(torch.load("models/src_informative/best_informative_model.pth", map_location=device))
informative_model.to(device)
informative_model.eval()

# --- Class mappings
HUMANITARIAN_LABELS = {0: "not_humanitarian", 1: "rescue_volunteering_or_donation_effort",
    2: "infrastructure_and_utilities_damage", 3: "affected_individuals", 4: "injured_or_dead_people",
    5: "missing_or_found_people", 6: "other_relevant_information"}

INFORMATIVE_LABELS = {0: "not_informative", 1: "informative"}


def predict(text: str, image_bytes: bytes):
    # --- Text encoding
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # --- Image preprocessing
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # --- Step 1: Informative prediction
    with torch.no_grad():
        informative_logits = informative_model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
        informative_pred = torch.argmax(informative_logits, dim=1).item()
        informative_label = INFORMATIVE_LABELS[informative_pred]

    # --- Step 2: Humanitarian prediction (if applicable)
    humanitarian_label = None
    if informative_label == "informative":
        with torch.no_grad():
            hum_logits = humanitarian_model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
            hum_pred = torch.argmax(hum_logits, dim=1).item()
            humanitarian_label = HUMANITARIAN_LABELS[hum_pred]

    return {"informative": informative_label, "humanitarian": humanitarian_label}
