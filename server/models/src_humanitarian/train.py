import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchvision import transforms
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from data_loader import HumanitarianDataset
from model import HumanitarianNetV1


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
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix for Best F1 Score')
        plt.tight_layout()
        figure.savefig(file_path)
        plt.close(figure)
    except Exception as e:
        print(f"Could not save confusion matrix: {e}")


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='best_humanitarian_model.pth'):
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
    TRAIN_CSV_PATH = '../data/processed_humanitarian/train.csv'
    VAL_CSV_PATH = '../data/processed_humanitarian/dev.csv'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HumanitarianDataset(TRAIN_CSV_PATH, IMAGE_DIR, train_transform)
    val_dataset = HumanitarianDataset(VAL_CSV_PATH, IMAGE_DIR, val_transform)

    num_workers = 0  # Debug mode: Avoid DataLoader deadlocks
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = HumanitarianNetV1(
        num_classes=config['num_classes'],
        unfreeze_bert_layers=config['unfreeze_bert_layers'],
        unfreeze_resnet_layers=config['unfreeze_resnet_layers']
    ).to(device)

    labels = [int(item['label']) for item in train_dataset]
    class_sample_count = np.bincount(labels, minlength=config['num_classes'])

    if np.any(class_sample_count == 0):
        print(f"⚠️ Warning: Classes with 0 samples: {np.where(class_sample_count == 0)[0]}")

    weight_per_class = 1. / (class_sample_count + 1e-6)
    weight_per_class = np.clip(weight_per_class, a_min=None, a_max=100)
    weights = torch.tensor(weight_per_class, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Using class weights: {weights.cpu().numpy()}")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

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

                if i % 10 == 0:
                    print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"🔥 Error during training batch: {e}")
            break

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation
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
        print("Classification Report:\n", classification_report(
            all_labels, all_preds, labels=expected_labels, target_names=class_names, zero_division=0
        ))

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_humanitarian_model.pth")
            print(f"New best F1 score: {best_f1:.4f}. Model saved.")
            cm = confusion_matrix(all_labels, all_preds, labels=expected_labels)
            plot_confusion_matrix(cm, class_names=class_names)

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered.")
            break

        torch.cuda.empty_cache()  # clean up memory after each epoch

    print("\n🎯 Training complete.")
    print(f"Best model saved at 'best_humanitarian_model.pth' with Val Loss: {early_stopping.val_loss_min:.4f}")
    print(f"Best Macro F1 Score: {best_f1:.4f}")


if __name__ == '__main__':
    hyperparameters = {
        'epochs': 25,
        'patience': 5,
        'batch_size': 16,
        'learning_rate': 5e-6,
        'unfreeze_bert_layers': 2,
        'unfreeze_resnet_layers': 2,
        'num_classes': 7,
        'seed': 42
    }
    train(hyperparameters)
