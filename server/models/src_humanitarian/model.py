import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class HumanitarianNetV1(nn.Module):
    def __init__(self, num_classes=7, unfreeze_bert_layers=2, unfreeze_resnet_layers=2):
        super(HumanitarianNetV1, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        if unfreeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[-unfreeze_bert_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        if unfreeze_resnet_layers > 0:
            resnet_children = list(self.resnet.children())
            for child in resnet_children[-(unfreeze_resnet_layers + 1):-1]:
                for param in child.parameters():
                    param.requires_grad = True

        resnet_out_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        bert_hidden_size = self.bert.config.hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(bert_hidden_size + resnet_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state[:, 0, :]
        image_features = self.resnet(image)
        combined_features = torch.cat((text_features, image_features), dim=1)
        output = self.fusion(combined_features)
        return output
