from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from torch import nn
import time



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






model_id = ".venv/base"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForVision2Seq.from_pretrained('./training/Flickr30k_1')
#model = AutoModelForVision2Seq.from_pretrained('./.venv/base')
#model = AutoModelForVision2Seq.from_pretrained('./training/caption')

model.eval()
#replace_blip_patch_with_resnet(model)
#print(model)

images = [Image.open(".venv/Flick/Images/301246.jpg")]

time_start = time.time()
processed = processor(images=images, padding="max_length", return_tensors="pt")
generated_output = model.generate(pixel_values=processed['pixel_values'], max_new_tokens=64)

print(time.time() - time_start)
print(processor.batch_decode(generated_output, skip_special_tokens=True))
