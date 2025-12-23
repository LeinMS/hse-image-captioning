# '''Our filtering is fine. The inputs_embeds are not manually passed; they are automatically included by PEFT.'
#
# PEFT currently has incomplete support for V+L structure models like BLIP, and errors may occur when wrapping forward.
#
# This is why you won't find many practical examples of LoRA fine-tuning BLIP1 Large on the community and GitHub—it's not as naturally compatible with LLM/RoBERTa/BERT as LoRA is with LLM/RoBERTa/BERT.
#
# If you just want BLIP1 Large to fine-tune your task, it is strongly recommended to perform full fine-tuning directly (not LoRA).
#
# If you are committed to LoRA, you might consider:
#
# Inject LoRA only into the visual part (excluding the text part), or #
# This approach involves injecting only the visual transformer using the LoRA Adapter, while leaving the text decoder unchanged. This requires custom code and cannot be done directly using the Peft toolchain.
#
# You can also try posting an issue on the github/peft repository to seek official support, explaining your bug and needs.
#
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
import os
from datasets import load_dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from PIL import Image
#
# # ==== User-modifiable configuration section ====
MODEL_NAME = "./.venv/base/"
JSONL_PATH = "./dataset/annotations.jsonl"
IMG_ROOT = "./dataset/"
OUTPUT_DIR = "./lora_blip_output"
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
MAX_SEQ_LENGTH = 32
# # ================================
#
#
class FilteredTrainer(Trainer):
    """
    Custom Trainer: Override compute_loss，
    1. Only retain the pixel values and labels that the model actually needs.，
    2. It also prints all the fields passed in the current batch and the fields
    that are actually passed to the model after filtering, which is convenient for debugging.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # First, print all fields passed in to the current batch.
        print(f"\n>> Fields passed to the Trainer in the current batch: {list(inputs.keys())}")
        # Only keep the two fields that BLIP1 forward actually needs: pixel_values and labels.
        filtered_inputs = {}
        if "pixel_values" in inputs:
            filtered_inputs["pixel_values"] = inputs["pixel_values"]
        if "labels" in inputs:
            filtered_inputs["labels"] = inputs["labels"]
        # Print it again to confirm which fields the model ultimately receives.
        print(f">> The fields that are actually passed to the model after filtering: {list(filtered_inputs.keys())}")
        # Call the model and calculate the loss.
        outputs = model(**filtered_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
def main():
    # 1. Load datasets in JSON format (one per line) {"image": "...", "text": "..."}）
    dataset = load_dataset("json", data_files=JSONL_PATH, split="train")
    # 2. Load the Processor and base model corresponding to BLIP1 Large.
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    base_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    print(base_model)
    # 3. Configure the LoRA parameters (using "self_attn.qkv" and "self_attn.projection" to match all layers).
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=[
            "self_attn.qkv",        # Matching all layers of the visual encoder's qkv linear layer
            "self_attn.projection", # Matching the projection of all layers in the visual encoder (linear layer)
            # To fine-tune the Text Decoder's Query/Value simultaneously, open the following two lines：
            # "attention.self.query",
            # "attention.self.value",
        ]
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Output all trainable LoRA parameters to confirm that it has taken effect.
    # 4. Define a preprocessing function for a single sample: load the image and process it along with the corresponding text.
    def preprocess(example):
        image_path = os.path.join(IMG_ROOT, example["image"])
        image = Image.open(image_path).convert("RGB")
        # Use a processor to process both images and text simultaneously.
        proc_outputs = processor(
            images=image,
            text=example["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        # Only pixel values are retained here. Input IDs or attention mask are no longer explicitly retrieved.
        inputs = {}
        inputs["pixel_values"] = proc_outputs["pixel_values"].squeeze(0)
        # Copy the input_ids and use it as labels; the model will automatically generate decoder_input_ids.
        inputs["labels"] = proc_outputs["input_ids"].squeeze(0).clone()
        return inputs
    # 5. Perform map preprocessing on the entire dataset to generate tensors that the model can directly input.
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        num_proc=1  # If the CPU has many cores, you can set it to 2-4 to improve speed.
    )
    # 6. Configure training parameters
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,                       # Path for saving weights after fine-tuning
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        save_strategy="epoch",                       # Save once per epoch
        save_total_limit=1,                          # A maximum of 1 checkpoint can be retained.
        logging_steps=10,                            # Print a log every 10 steps.
        fp16=True if torch.cuda.is_available() else False,
        report_to="none"                             # Disable wandb log reporting, etc.
    )
    # 7. Create a custom trainer and start training.
    trainer = FilteredTrainer(
        model=model,
        args=args,
        train_dataset=dataset
    )
    print("\n>>> The LoRA Adapter parameters are as follows (trainable parameters):")
    model.print_trainable_parameters()
    print("\n>>> Start LoRA fine-tuning training ...")
    trainer.train()
    # 8. Save the LoRA-tuned weights after training.
    model.save_pretrained(OUTPUT_DIR)
    print(f"\n>>> Training is complete, and the LoRA-tuned model weights have been saved：{OUTPUT_DIR}")
if __name__ == "__main__":
    main()