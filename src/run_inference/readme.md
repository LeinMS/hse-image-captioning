# Qwen3-VL-2B Fine-Tuned Model Inference Test

This .ipynb file contains a short script for testing a fine-tuned vision-language model (`Qwen3-VL-2B-Instruct-trl-sft`) on a dataset that was *not* used during its additional training.

The goal of this test is to evaluate how the model performs on unseen image data by generating a textual description of a randomly selected image.

## Overview

The notebook script performs the following steps:

1. **Environment Setup**
   - Installs required libraries such as `transformers`, `accelerate`, `flash-attn`, and `datasets`.
   - Checks GPU availability.

2. **Model Loading**
   - Loads the base Qwen3-VL-2B vision-language model.
   - Applies a LoRA adapter (`leinms/Qwen3-VL-2B-Instruct-trl-sft`) for fine-tuned weights.
   - Loads the corresponding processor for preparing inputs.

3. **Dataset**
   - Loads a subset (`train[:10%]`) of the `bodhisattamaiti/StylExNet5k` dataset from Hugging Face.
   - This dataset was not part of the model’s fine-tuning, making it suitable for an out-of-distribution test.

4. **Inference Function**
   - The `caption_with_qwen(example)` function:
     - Takes a dataset item (including a PIL image).
     - Formats the prompt with an image and a text request using the processor.
     - Runs the model to generate a text description (“Describe this image”).
     - Decodes and returns the generated answer.

5. **Random Sampling and Display**
   - The `show_and_caption_random(ds)` function:
     - Chooses a random index from the dataset.
     - Displays the associated image.
     - Calls the captioning function to generate and return a description.
