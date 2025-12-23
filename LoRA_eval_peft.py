from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from torch import nn

model_id = ".venv/base"

processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForVision2Seq.from_pretrained('./training/caption')
model.eval()
#print(model)

images = [Image.open("dataset/image_train/blue_on_green_0108.jpg")]

processed = processor(images=images, padding="max_length", return_tensors="pt")
generated_output = model.generate(pixel_values=processed['pixel_values'], max_new_tokens=64)

print(processor.batch_decode(generated_output, skip_special_tokens=True))
