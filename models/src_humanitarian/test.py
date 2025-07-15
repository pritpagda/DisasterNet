import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from data_loader import HumanitarianDataset
from model import HumanitarianNetV1

def plot_confusion_matrix(cm, class_names, file_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 8), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()   # Display the confusion matrix window
    plt.close()

def test_model(csv_path, image_dir, model_path, batch_size=16, num_classes=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = HumanitarianDataset(csv_path, image_dir, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    model = HumanitarianNetV1(
        num_classes=num_classes,
        unfreeze_bert_layers=0,
        unfreeze_resnet_layers=0
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"Test Macro F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    print(f"Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    TEST_CSV_PATH = '../data/processed_humanitarian/test.csv'
    IMAGE_DIR = '../data/'
    MODEL_PATH = 'best_f1_model.pth'

    test_model(TEST_CSV_PATH, IMAGE_DIR, MODEL_PATH, batch_size=16, num_classes=7)
