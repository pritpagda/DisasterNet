import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup

from data_loader import DamageDataset
from model import DamageNetV1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm, class_names, file_path="best_confusion_matrix.png"):
    try:
        figure = plt.figure(figsize=(8, 8), dpi=100)
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix for Best F1 Score')
        plt.tight_layout()
        figure.savefig(file_path)
        plt.close(figure)
    except Exception as e:
        print(f"Could not save confusion matrix: {e}")


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='best_damage_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(config):
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMAGE_DIR = '../data/'
    TRAIN_CSV_PATH = '../data/processed_damage/train.csv'
    VAL_CSV_PATH = '../data/processed_damage/dev.csv'

    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    train_dataset = DamageDataset(TRAIN_CSV_PATH, IMAGE_DIR, train_transform)
    val_dataset = DamageDataset(VAL_CSV_PATH, IMAGE_DIR, val_transform)

    num_workers = 0
    pin_memory = torch.cuda.is_available()

    # REMOVED the WeightedRandomSampler for now to improve stability
    # Using simple shuffling instead.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
        # Using shuffle instead of a sampler
        num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    model = DamageNetV1(num_classes=config['num_classes'], unfreeze_bert_layers=config['unfreeze_bert_layers'],
                        unfreeze_resnet_layers=config['unfreeze_resnet_layers']).to(device)

    # We will still use weighted loss, as it's a stable way to handle imbalance
    labels = [int(item['label']) for item in train_dataset]
    class_sample_count = np.bincount(labels, minlength=config['num_classes'])
    weight_per_class = 1. / (class_sample_count + 1e-6)
    weights = torch.tensor(weight_per_class, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Using class weights in loss function: {weights.cpu().numpy()}")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    print(f"Successfully created optimizer with single learning rate: {config['learning_rate']}")

    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    best_f1 = 0.0

    for epoch in range(config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")
        model.train()
        total_train_loss = 0

        try:
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                with autocast(device_type=device.type):
                    outputs = model(input_ids, attention_mask, images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_train_loss += loss.item()

        except Exception as e:
            print(f"🔥 Error during training batch: {e}")
            break

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    with autocast(device_type=device.type):
                        outputs = model(input_ids, attention_mask, images)
                        loss = criterion(outputs, labels)

                    total_val_loss += loss.item()
                    _, preds = torch.max(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"🔥 Error in validation batch {j}: {e}")
                    continue

        avg_val_loss = total_val_loss / len(val_loader)

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        class_names = [f"class_{i}" for i in range(config['num_classes'])]
        expected_labels = list(range(config['num_classes']))

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Macro F1: {f1:.4f}")
        print("Classification Report:\n",
              classification_report(all_labels, all_preds, labels=expected_labels, target_names=class_names,
                                    zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_f1_model.pth")
            print(f"New best F1 score: {best_f1:.4f}. Model saved to 'best_f1_model.pth'.")
            cm = confusion_matrix(all_labels, all_preds, labels=expected_labels)
            plot_confusion_matrix(cm, class_names=class_names)

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

        torch.cuda.empty_cache()

    print("\n🎯 Training complete.")
    print(f"Best model (by val loss) saved at '{early_stopping.path}' with Val Loss: {early_stopping.val_loss_min:.4f}")
    print(f"Best model (by F1 score) saved at 'best_f1_model.pth' with Macro F1 Score: {best_f1:.4f}")


if __name__ == '__main__':
    hyperparameters = {'epochs': 15, 'patience': 3, 'batch_size': 16, 'learning_rate': 1e-5,
        'unfreeze_bert_layers': 1,  # Freezing almost all layers for stability
        'unfreeze_resnet_layers': 1,  # Freezing almost all layers for stability
        'num_classes': 3, 'seed': 42}
    train(hyperparameters)
