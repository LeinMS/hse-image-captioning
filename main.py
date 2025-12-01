from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
#import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the Processor and Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the Image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Prepare the Inputs
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

# Generate the Caption
output = model.generate(**inputs)
print(f"Description : {processor.decode(output[0], skip_special_tokens=True)}")