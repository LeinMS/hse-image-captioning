import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
from lora_layer import LoRALinear
from model_patch import patch_blip1_visual_transformer

# =====
MODEL_NAME = "./.venv/base/"
LORA_PATH = "./lora_blip_output/lora_bs2_ep5_lr0.0001_r16_a32_20251223_214940.pth"
IMAGE_PATH = "./dataset/image_train/green_on_blue_0097.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====
print("Loading the BLIP model...")
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
processor = BlipProcessor.from_pretrained(MODEL_NAME)




import torch
import torch.nn as nn
from torchvision.models import resnet18
class ResNetPatchEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        backbone = resnet18()
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
        self.out_channels = backbone.layer4[-1].conv2.out_channels  # 512 для resnet18
        self.proj = nn.Conv2d(self.out_channels, hidden_size, 3, 3, 1)
        self._first_conv = backbone.conv1

    @property
    def weight(self):
        return self._first_conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.proj(x)
        #print(x.shape)
        x = x.flatten(2)

        return x

def replace_blip_patch_with_resnet(blip_model):
    hidden_size = blip_model.config.vision_config.hidden_size
    resnet_patch = ResNetPatchEmbedding(hidden_size=hidden_size)

    if hasattr(blip_model.vision_model, "embeddings") and \
       hasattr(blip_model.vision_model.embeddings, "patch_embedding"):
        blip_model.vision_model.embeddings.patch_embedding = resnet_patch
    elif hasattr(blip_model.vision_model, "patch_embed"):
        blip_model.vision_model.patch_embed = resnet_patch
    else:
        raise RuntimeError("Cannot find patch embedding module in BLIP vision_model")

replace_blip_patch_with_resnet(model)
print(model.vision_model)






# =========== Patch LoRA ===========
print("Apply LoRA patch to the visual backbone...")
patch_blip1_visual_transformer(model, r=16, alpha=32, dropout=0.05, verbose=True)

# =========== Load LoRA weights ===========
print(f"Load LoRA Adapter weights: {LORA_PATH}")
lora_state = torch.load(LORA_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(lora_state, strict=False)
if len(missing) > 0:
    print("[WARNING] Some parameters were not loaded.", missing)
if len(unexpected) > 0:
    print("[WARNING] There are extra parameters:", unexpected)
print(model)
model = model.to(DEVICE)
model.eval()

# ====
# Example: Generating a caption for an image
image = Image.open(IMAGE_PATH).convert("RGB")
prompt = ""  # This can be empty, or a custom question.

inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    output_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"] if "input_ids" in inputs else None,
        attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
        max_new_tokens=30
    )
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("\n>> Generated description:", caption)