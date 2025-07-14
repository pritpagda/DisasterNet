import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


class DisasterNetV1(nn.Module):
    def __init__(self, num_classes=2, unfreeze_bert_layers=2, unfreeze_resnet_layers=2):
        """
        Initializes the DisasterNetV1 model.

        Args:
            num_classes (int): The number of output classes for classification.
            unfreeze_bert_layers (int): Number of final BERT encoder layers to unfreeze.
            unfreeze_resnet_layers (int): Number of final ResNet blocks to unfreeze.
        """
        super(DisasterNetV1, self).__init__()

        # --- Text Encoder (BERT) ---
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze all BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
        # Unfreeze the last 'n' layers of the BERT encoder for fine-tuning
        if unfreeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[-unfreeze_bert_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # --- Image Encoder (ResNet) ---
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Freeze all ResNet parameters initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze the last 'n' residual blocks of ResNet (e.g., layer4, layer3)
        if unfreeze_resnet_layers > 0:
            # ResNet blocks are children of the main model module
            # We access them sequentially from the end to unfreeze layer4, then layer3, etc.
            resnet_children = list(self.resnet.children())
            for child in resnet_children[- (unfreeze_resnet_layers + 1): -1]:  # +1 to include the last block layer
                for param in child.parameters():
                    param.requires_grad = True

        # Get the number of input features for the ResNet's classifier
        resnet_out_features = self.resnet.fc.in_features
        # Replace the final classification layer with an Identity layer to get features
        self.resnet.fc = nn.Identity()

        # --- Fusion and Classifier Head ---
        bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for bert-base

        # This head takes the concatenated features from both backbones
        self.fusion = nn.Sequential(
            nn.Linear(bert_hidden_size + resnet_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        """
        Defines the forward pass of the model.

        Args:
            input_ids: Tensor of token ids from the tokenizer.
            attention_mask: Tensor indicating which tokens to attend to.
            image: Tensor representing the input image.

        Returns:
            The final classification logits.
        """
        # 1. Process Text with BERT
        # The output of the [CLS] token is used as the text's feature representation
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # 2. Process Image with ResNet
        image_features = self.resnet(image)  # Shape: (batch_size, 2048)

        # 3. Concatenate (Fuse) Features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # 4. Final Classification
        output = self.fusion(combined_features)

        return output