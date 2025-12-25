from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import os, json, torch
from PIL import Image
import requests
device = "cuda" if torch.cuda.is_available() else "cpu"






import torch
import torch.nn as nn
class ResNetPatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torch.hub.load('pytorch/vision', 'resnet18')
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.proj = nn.Conv2d(512, 768, 1, 1, 0)
        self.fc = nn.Linear(768, 10)
        self._first_conv = backbone.conv1

    @property
    def weight(self):
        return self._first_conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.proj(x)
        # print(x.shape)
        x = x.flatten(2)

        return x

def replace_blip_patch_with_resnet(blip_model):
    hidden_size = blip_model.config.vision_config.hidden_size
    resnet_patch = ResNetPatchEmbedding()
    del resnet_patch.fc
    resnet_patch.load_state_dict(torch.load('model_resnet.pth', weights_only=True), strict=False)

    if hasattr(blip_model.vision_model, "embeddings") and \
            hasattr(blip_model.vision_model.embeddings, "patch_embedding"):
        blip_model.vision_model.embeddings.patch_embedding = resnet_patch
    elif hasattr(blip_model.vision_model, "patch_embed"):
        blip_model.vision_model.patch_embed = resnet_patch
    else:
        raise RuntimeError("Cannot find patch embedding module in BLIP vision_model")








config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "self.query",
        "self.key",
        "self.value",
        "output.dense",
        "self_attn.qkv",
        "self_attn.projection",
        "mlp.fc1",
        "mlp.fc2",
    ],
)

model_path = "./.venv/base/"
model = AutoModelForVision2Seq.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)


#replace_blip_patch_with_resnet(model)


model = get_peft_model(model, config).to(device)
model.print_trainable_parameters()


from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import Counter, defaultdict

class FlickrCSV(Dataset):
    def __init__(self, image_root: str, captions_csv: str, processor, transform=None):
        self.processor, self.image_root = processor, image_root
        self.transform = transform

        # Read CSV (handles .csv or .txt with CSV content)
        df = pd.read_csv(captions_csv)
        if not {"image", "caption"}.issubset(df.columns):
            raise ValueError("captions file must have columns: 'image','caption'")

        # keep only entries whose image file exists
        self.items = []
        for img, cap in zip(df["image"], df["caption"]):
            img_path = os.path.join(self.image_root, img)
            if os.path.exists(img_path):
                self.items.append((img, str(cap)))


        if len(self.items) == 0:
            raise RuntimeError(
                f"No valid (image, captions) pairs found. "
                f"Check paths:\n image_root={image_root}\n captions_csv={captions_csv}\n"
                f"and ensure header is 'image,caption'."
            )


    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        img_name, caps = self.items[idx]
        path = os.path.join(self.image_root, img_name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        proc = self.processor(
            images=image, text=caps,
            padding="max_length", max_length=None, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in proc.items()}
        item["labels"] = item["input_ids"].clone()
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}




from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(2),
    transforms.RandomVerticalFlip(2),
    transforms.RandomRotation(10)
])

train_dataset   = FlickrCSV("./.venv/Flick/Images", "./.venv/Flick/captions.txt", processor, transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collate_fn)

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

model.train()


for epoch in range(1):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        loss = outputs.loss

        print("idx:", idx, "Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if idx % 10 == 0:
            generated_output = model.generate(pixel_values=pixel_values, max_length=100)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))

model.save_pretrained('./training/caption_new')
