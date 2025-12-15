# Fine-Tuning Qwen3-VL-2B with Supervised Fine-Tuning (SFT)

This folder contains a Jupyter Notebook that demonstrates supervised fine-tuning (SFT) of the **Qwen3-VL-2B** vision-language model using parameter-efficient techniques. The workflow focuses on adapting the model to image–text captioning data while minimizing GPU memory usage through LoRA and quantization.

## Overview

The notebook performs end-to-end fine-tuning of a multimodal large language model that accepts both images and text as inputs. The training pipeline is built on top of the Hugging Face ecosystem and leverages:

- **Qwen3-VL-2B** as the base vision-language model
- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
- **TRL (Transformer Reinforcement Learning)** SFT utilities
- **Hugging Face Datasets and Hub** for data loading and model publishing

The result is a fine-tuned multimodal model suitable for image captioning and related vision–language tasks.

## Dataset

The model is fine-tuned using the following dataset:

- **Mozilla/flickr30k-transformed-captions-gpt4o**

This dataset is derived from Flickr30k and contains image–caption pairs, where captions have been transformed or enhanced using GPT-4o. The notebook loads the dataset via the Hugging Face `datasets` library and uses it directly for supervised fine-tuning.

## Key Technologies and Libraries

- `transformers`
- `trl` (with PEFT support)
- `peft`
- `bitsandbytes`
- `datasets`
- `torch`
- `huggingface_hub`

## Training Approach

### Model Loading
- The base **Qwen3-VL-2B** model is loaded with 4-bit quantization enabled.
- Gradient checkpointing is used to further reduce memory usage.

### Parameter-Efficient Fine-Tuning
- LoRA adapters are applied to selected attention and projection layers.
- Only LoRA parameters are trained, while the base model weights remain frozen.

### Data Processing
- Each training sample consists of:
  - An image
  - A corresponding caption formatted as a supervised instruction/response pair
- Inputs are tokenized using the Qwen VL processor to properly handle multimodal data.

### Training Configuration
- Supervised fine-tuning is performed using `SFTTrainer`
- Mixed precision training (FP16/BF16 depending on hardware)
- Training statistics such as runtime and GPU memory usage are tracked

## Memory Efficiency

The notebook explicitly measures:
- Peak GPU memory usage
- Additional memory consumed by LoRA adapters
- Percentage of total GPU memory utilized

This makes the setup suitable for fine-tuning large vision-language models on consumer or limited GPUs.

## Output and Model Publishing

After training:
- The fine-tuned model is saved locally
- The model (including LoRA adapters) is pushed to the Hugging Face Hub
- The original dataset name is preserved in the model card metadata

## Notebook Structure

The notebook is organized into the following logical steps:

1. Environment setup and dependency installation
2. Hugging Face authentication
3. Dataset loading
4. Model and processor initialization
5. LoRA configuration
6. Supervised fine-tuning with `SFTTrainer`
7. Training metrics and GPU memory reporting
8. Model saving and Hub upload

## Requirements

- Python 3.9+
- CUDA-enabled GPU recommended
- Hugging Face account for model upload




