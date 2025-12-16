# Inference with Fine-Tuned Qwen3-VL-2B (Vision-Language)

This folder contains a Jupyter Notebook that demonstrates **inference and evaluation** of a fine-tuned **Qwen3-VL-2B-Instruct** vision-language model. The notebook is designed to load a previously fine-tuned model (via SFT with TRL and LoRA) and perform multimodal generation using image–text inputs.

## Overview

The notebook focuses exclusively on **model inference**, not training. It shows how to:

- Load a fine-tuned Qwen3-VL-2B model
- Properly initialize the multimodal processor
- Run vision–language generation on image inputs
- Decode and display model outputs
- Use memory-efficient inference settings

The workflow is suitable for testing, validation, and downstream application development.

## Model

- **Base architecture:** Qwen3-VL-2B-Instruct
- **Fine-tuning method:** Supervised Fine-Tuning (SFT) with TRL
- **Adaptation technique:** LoRA (parameter-efficient fine-tuning)
- **Precision:** FP16 / BF16 (hardware-dependent)

The notebook assumes the model has already been fine-tuned and is either:
- Stored locally, or
- Available on the Hugging Face Hub

## Key Libraries

- `transformers`
- `torch`
- `peft`
- `PIL`
- `huggingface_hub`

## Inference Pipeline

### Model and Processor Loading
- The fine-tuned Qwen3-VL-2B model is loaded using `AutoModelForVision2Seq`
- The corresponding `AutoProcessor` is used to correctly handle image and text inputs
- LoRA adapters are automatically applied during loading

### Input Format
Each inference example consists of:
- An image (loaded via PIL)
- A text prompt or instruction compatible with the Qwen Instruct format

The notebook demonstrates how to structure multimodal inputs so that both vision and language tokens are processed correctly.

### Generation
- Text generation is performed using `model.generate`
- Common decoding parameters include:
  - `max_new_tokens`
  - `do_sample`
  - `temperature`
  - `top_p`
- Outputs are decoded into readable text using the processor

## Memory and Performance Considerations

The notebook is optimized for efficient inference:
- Supports GPU acceleration
- Compatible with quantized models
- Avoids unnecessary gradient tracking
- Can be executed on limited-memory GPUs

## Notebook Structure

1. Environment and dependency setup
2. Model and processor loading
3. Image loading and preprocessing
4. Prompt construction
5. Multimodal inference
6. Output decoding and visualization

## Requirements

- Python 3.9+
- CUDA-enabled GPU recommended for fast inference
- Hugging Face account 
