# train_lora_blip_manual.py  (Keep only this one copy as the final version)
import os, json, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    get_cosine_schedule_with_warmup,  # Cosine Scheduler
)

from lora_layer import LoRALinear
from model_patch import patch_blip1_visual_transformer

# ====
MODEL_NAME   = "Salesforce/blip-image-captioning-base"
JSONL_PATH   = "./dataset/annotations.jsonl"
IMAGE_ROOT   = "./dataset/"
OUTPUT_DIR   = "./lora_blip_output"

BATCH_SIZE   = 2
NUM_EPOCHS   = 50
LR           = 1e-4

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ====
USE_FP16          = True          # switch FP16
GRAD_ACCUM_STEPS  = 4             # Accumulate n small batches, then step → equivalent large batch
WARMUP_STEPS      = 100           # Linear warm-up steps

# =====
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

# =====
def main():
    print(f"Loading Model: {MODEL_NAME}")
    model      = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor  = BlipProcessor.from_pretrained(MODEL_NAME)

    # LoRA Patch
    print(f"Apply LoRA Patch to the visual backbone（r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}）")
    patch_blip1_visual_transformer(
        model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, verbose=True
    )

    # Freeze non-LoRA parameters
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n or "lora_B" in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable):,}")

    # Data loading
    dataset   = BlockDataset(JSONL_PATH, IMAGE_ROOT, processor)
    dataloader= DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, collate_fn=collate_fn)

    # ================== Optimizer & Scheduler ==================
    optimizer  = torch.optim.AdamW(trainable, lr=LR)
    total_steps= (len(dataloader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS
    scheduler  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    print(f"Total number of optimization steps: {total_steps}  |  warm-up 步数: {WARMUP_STEPS}")

    # AMP settings
    if USE_FP16 and DEVICE == "cuda":
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print(">> Mixed precision is enabled. (FP16)")
    else:
        scaler = None

    # ================== Training loop ==================
    model.to(DEVICE)
    model.train()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader)):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids    = batch["input_ids"].to(DEVICE)
            attn_mask    = batch["attention_mask"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            if scaler:  # FP16
                with autocast():
                    out  = model(pixel_values=pixel_values,
                                 input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 labels=labels)
                    loss = out.loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:       # FP32
                out  = model(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attn_mask,
                             labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS
                loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            # === Gradients are accumulated to a specified number of steps before being updated. ===
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")

    # ================== Save LoRA weights ==================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"lora_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_lr{LR}_r{LORA_R}_a{LORA_ALPHA}_{ts}.pth"
    save_path = os.path.join(OUTPUT_DIR, name)
    torch.save({k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora_A" in k or "lora_B" in k},
               save_path)
    print(f"[✓] LoRA Adapter Saved to: {save_path}")

if __name__ == "__main__":
    main()