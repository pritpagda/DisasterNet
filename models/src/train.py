import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_loader import CrisisMMDDataset
from model import DisasterNetV1


def plot_confusion_matrix(cm, class_names):
    """Returns a matplotlib figure containing the plotted confusion matrix."""
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    return figure


def train():
    # --- 1. W&B Initialization ---
    wandb.init(project="disasternet-v1")
    config = wandb.config

    # --- 2. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CORRECTED PATHS ---
    # Since train.py is in 'src', we go up one level ('../') to get to 'models/'
    IMAGE_DIR = '../data/'
    TRAIN_CSV_PATH = '../data/processed/train.csv'
    VAL_CSV_PATH = '../data/processed/dev.csv'
    # --- END CORRECTIONS ---

    # --- 3. Data Loading ---
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CrisisMMDDataset(TRAIN_CSV_PATH, IMAGE_DIR, tokenizer, image_transform)
    val_dataset = CrisisMMDDataset(VAL_CSV_PATH, IMAGE_DIR, tokenizer, image_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- 4. Model Initialization ---
    model = DisasterNetV1(
        num_classes=2,
        unfreeze_bert_layers=config.unfreeze_bert_layers,
        unfreeze_resnet_layers=config.unfreeze_resnet_layers
    ).to(device)

    wandb.watch(model, log="all")

    # --- 5. Optimizer and Loss Function ---
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # --- 6. Training & Validation Loop ---
    best_f1 = 0.0

    for epoch in range(config.epochs):
        # -- Training --
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation --
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # -- Calculate Metrics --
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        print(
            f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f}")

        # -- W&B Logging --
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": f1,
            "val_precision": precision,
            "val_recall": recall
        })

        # -- Save Best Model and Log Confusion Matrix --
        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1 score: {best_f1:.4f}. Saving model...")

            model_save_path = f"best_model_.pth"
            torch.save(model.state_dict(), model_save_path)

            cm = confusion_matrix(all_labels, all_preds)
            cm_plot = plot_confusion_matrix(cm, class_names=['not_informative', 'informative'])
            wandb.log({"confusion_matrix": wandb.Image(cm_plot)})
            plt.close(cm_plot)

            #artifact = wandb.Artifact('DisasterNet-v1', type='model')
            #artifact.add_file(model_save_path)
            #wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    hyperparameters = {
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 5e-6,
        'bert_model_name': 'bert-base-uncased',
        'unfreeze_bert_layers': 2,
        'unfreeze_resnet_layers': 2,
    }

    wandb.init(config=hyperparameters)
    train()