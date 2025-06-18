import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
import open_clip

class CLIPLoRADetector(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", lora_r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        for param in self.clip.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.clip.text_model = get_peft_model(self.clip.text_model, lora_config)
        self.clip.text_model.print_trainable_parameters()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=dummy)
            feat_dim = vision_outputs.pooler_output.shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1)
        )

    def forward(self, images):
        vision_outputs = self.clip.vision_model(pixel_values=images)
        image_embeds = vision_outputs.pooler_output
        logits = self.classifier(image_embeds).squeeze(1)
        return logits


    def get_processor(self):
        return self.processor


class CLIPLinearProbe(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='openai', embed_dim=512):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.clip.encode_image(x)
        return self.linear(features)