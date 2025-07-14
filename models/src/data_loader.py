import os
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class CrisisMMDDataset(Dataset):
    def __init__(self, annotations_file, img_dir, tokenizer, transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Base directory for constructing the full image path.
            tokenizer: A BERT tokenizer instance.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # --- UPDATED TO MATCH YOUR CSV ---
        # 1. Get Text and Label
        # Use the 'text' column for text and 'label' for the direct integer label.
        text = self.annotations_df.loc[idx, 'text']
        label = self.annotations_df.loc[idx, 'label']

        # 2. Get Image
        # The 'image_path' column now contains the relative path to the image.
        # We join it with the base image directory provided.
        img_relative_path = self.annotations_df.loc[idx, 'image_path']
        img_path = os.path.join(self.img_dir, img_relative_path)
        # --- END OF UPDATE ---

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"Warning: Could not load image {img_path}. Using a dummy image.")
            image = Image.new('RGB', (224, 224), color='red')

        # 3. Tokenize Text
        text_encoding = self.tokenizer(
            str(text), # Ensure text is a string
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 4. Apply Image Transforms
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }