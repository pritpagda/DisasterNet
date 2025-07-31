import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup

import wandb
from data_loader import InformativeDataset
from model import InformativeNet


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    return figure


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def train(hyperparameters):
    wandb.init(project="Informative", config=hyperparameters)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMAGE_DIR = '../data/'
    TRAIN_CSV_PATH = '../data/processed_informative/train.csv'
    VAL_CSV_PATH = '../data/processed_informative/dev.csv'

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    train_dataset = InformativeDataset(TRAIN_CSV_PATH, IMAGE_DIR, train_transform)
    val_dataset = InformativeDataset(VAL_CSV_PATH, IMAGE_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = InformativeNet(num_classes=config.num_classes, unfreeze_bert_layers=config.unfreeze_bert_layers,
                           unfreeze_resnet_layers=config.unfreeze_resnet_layers).to(device)

    wandb.watch(model, log="all")

    # Calculate class weights if dataset is imbalanced
    labels = [item['label'].item() for item in train_dataset]
    class_sample_count = np.bincount(labels)
    weight_per_class = 1. / (class_sample_count + 1e-6)
    weights = torch.tensor(weight_per_class, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    scaler = GradScaler()

    best_f1 = 0.0
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

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

                with autocast():
                    outputs = model(input_ids, attention_mask, images)
                    loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1": f1,
            "val_precision": precision, "val_recall": recall})

        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1 score: {best_f1:.4f}. Saving model...")

            torch.save(model.state_dict(), "best_informative_model.pth")

            cm = confusion_matrix(all_labels, all_preds)
            cm_plot = plot_confusion_matrix(cm, class_names=['class_0', 'class_1'])  # update names if needed
            wandb.log({"confusion_matrix": wandb.Image(cm_plot)})
            plt.close(cm_plot)

        if early_stopping(f1):
            print("Early stopping triggered")
            break

    wandb.finish()


if __name__ == '__main__':
    hyperparameters = {'epochs': 15, 'batch_size': 16, 'learning_rate': 1e-5, 'bert_model_name': 'bert-base-uncased',
        'unfreeze_bert_layers': 4, 'unfreeze_resnet_layers': 3, 'num_classes': 2}
    train(hyperparameters)
