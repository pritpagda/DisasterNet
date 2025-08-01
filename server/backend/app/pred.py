import os
from io import BytesIO

import torch
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import BertTokenizer

from models.src_damage.model import DamageNetV1
from models.src_humanitarian.model import HumanitarianNetV1
from models.src_informative.model import InformativeNet

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

informative_model_path = hf_hub_download(repo_id="pritpagda/DisasterNet", filename="best_informative_model.pth",
                                         token=hf_token)
informative_model = InformativeNet(num_classes=2)
informative_model.load_state_dict(torch.load(informative_model_path, map_location=device))
informative_model.to(device)
informative_model.eval()

humanitarian_model_path = hf_hub_download(repo_id="pritpagda/DisasterNet", filename="best_humanitarian_model.pth",
                                          token=hf_token)
humanitarian_model = HumanitarianNetV1(num_classes=7)
humanitarian_model.load_state_dict(torch.load(humanitarian_model_path, map_location=device))
humanitarian_model.to(device)
humanitarian_model.eval()

damage_model_path = hf_hub_download(repo_id="pritpagda/DisasterNet", filename="best_damage_model.pth", token=hf_token)
damage_model = DamageNetV1(num_classes=3)
damage_model.load_state_dict(torch.load(damage_model_path, map_location=device))
damage_model.to(device)
damage_model.eval()

INFORMATIVE_LABELS = {0: "not_informative", 1: "informative"}

HUMANITARIAN_LABELS = {0: "not_humanitarian", 1: "rescue_volunteering_or_donation_effort",
                       2: "infrastructure_and_utilities_damage", 3: "affected_individuals", 4: "injured_or_dead_people",
                       5: "missing_or_found_people", 6: "other_relevant_information"}

DAMAGE_LABELS = {0: "severe_damage", 1: "mild_damage", 2: "little_or_no_damage"}


def predict(text: str, image_bytes: bytes):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        informative_logits = informative_model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
        informative_pred = torch.argmax(informative_logits, dim=1).item()
        informative_label = INFORMATIVE_LABELS[informative_pred]

    humanitarian_label = None
    damage_label = None

    if informative_label == "informative":
        with torch.no_grad():
            hum_logits = humanitarian_model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
            hum_pred = torch.argmax(hum_logits, dim=1).item()
            humanitarian_label = HUMANITARIAN_LABELS[hum_pred]

            damage_logits = damage_model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
            damage_pred = torch.argmax(damage_logits, dim=1).item()
            damage_label = DAMAGE_LABELS[damage_pred]

    return {"informative": informative_label, "humanitarian": humanitarian_label, "damage": damage_label,
            "text": text, }
