import os
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class HumanitarianDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.annotations_df.iloc[idx]
        text = str(row['text'])
        label = int(row['label'])
        img_relative_path = row['image_path']
        img_path = os.path.join(self.img_dir, img_relative_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            print(f"Warning: Could not load image {img_path}. Using a dummy image.")
            image = Image.new('RGB', (224, 224), color='red')

        text_encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
