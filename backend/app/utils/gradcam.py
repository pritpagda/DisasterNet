import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_ids, attention_mask, image_tensor, class_idx=None):
        self.model.eval()
        input_ids = input_ids.to(image_tensor.device)
        attention_mask = attention_mask.to(image_tensor.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, image=image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(self, image_pil, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        image_np = np.array(image_pil.resize((224, 224))).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = np.clip((1 - alpha) * image_np + alpha * heatmap_color, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
