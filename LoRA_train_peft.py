from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import os, json, torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

config = LoraConfig(
    r=16,
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

model = get_peft_model(model, config).to(device)
model.print_trainable_parameters()


from torch.utils.data import DataLoader, Dataset

class BlockDataset(Dataset):
    def __init__(self, jsonl_path, image_root, processor):
        self.items, self.processor, self.image_root = [], processor, image_root
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                img = data["image"] if "image" in data else data["image_path"]
                cap = data["caption"] if "caption" in data else data["text"]
                self.items.append((os.path.join(image_root, img), cap))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        proc = self.processor(
            images=image, text=caption,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in proc.items()}
        item["labels"] = item["input_ids"].clone()
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

train_dataset   = BlockDataset("./dataset/annotations.jsonl", "./dataset/", processor)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

model.train()


for epoch in range(10):
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
            generated_output = model.generate(pixel_values=pixel_values)
            print(processor.batch_decode(generated_output, skip_special_tokens=True))

model.save_pretrained('./training/caption')
