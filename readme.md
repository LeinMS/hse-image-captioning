# Study of Transformer Architectures and Fine-Tuning for Image Captioning

This project (HSE – Higher School of Economics) explores transformer-based vision–language models for the task of **image captioning**.  
The main focus is evaluating how different visual encoders and fine-tuning strategies influence model performance.

We use **Qwen/Qwen3-VL-2B-Instruct** as the base model and the **Flickr30k** dataset (`nlphuji/flickr30k`) for experiments.
HuggingFace link: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
---

## 📌 Project Goal

To study transformer architectures for image captioning, compare baseline and fine-tuned models, and analyze how fine-tuning improves caption quality.

---

# 📁 Project Stages

## 1. Dataset Preparation and Exploratory Analysis

- Collect and prepare the dataset (Flickr30k), optionally combining multiple sources.  
- Convert images and annotations into a unified format suitable for model training.  
- Perform initial analysis:
  - number and resolution of images  
  - object category statistics  
  - typical caption structures  
- **Deliverables:** short textual report or slides summarizing dataset properties.

---

## 2. Model Architecture Selection and Baseline Evaluation

- Select the primary architecture (**Qwen3-VL-2B-Instruct**) and optionally alternative lightweight VLMs for comparison.  
- Justify the choice using references to papers or benchmarks.  
- Run a **baseline evaluation** without fine-tuning to measure initial model performance.  
- **Deliverables:** brief rationale and links to sources.

---

## 3. Error Analysis

- Examine where the baseline model fails and why.  
- Collect statistics on:
  - false positives  
  - false negatives  
  - misidentified or missing objects  
- Use a **strong external evaluator (OpenAI GPT-5.1 API)** to:
  - compare generated captions with ground truth  
  - identify semantic errors  
  - detect omissions and hallucinations  
- Visualize correct and incorrect predictions for qualitative analysis.  
- **Deliverables:** structured summary of common errors and GPT-based evaluation insights.

---

## 4. Model Training and Fine-Tuning

- Fine-tune the selected model on the Flickr30k training set.  
- Evaluate on the public test split.  
- Record:
  - training curves (loss, LR, metrics)  
  - evaluation results on standard captioning metrics  
- **Deliverables:** screenshots, plots, or presentation summarizing results.

---

## 5. Baseline vs Fine-Tuned Model Comparison

- Compare performance of:
  - **baseline model (no fine-tuning)**  
  - **fine-tuned model on Flickr30k**  
- Use metrics such as BLEU, ROUGE, METEOR, CIDEr, SPICE.  
- Analyze improvements in:
  - object detection accuracy  
  - caption relevance  
  - error reduction (hallucinations, missed objects)  
- Provide visual examples showing how fine-tuning changes model outputs.  
- **Deliverables:** comparison tables, plots, qualitative examples, key insights.


# 📚 Dataset

**Flickr30k**  
HuggingFace link: https://huggingface.co/datasets/nlphuji/flickr30k

---

# 🧠 Model

**Qwen/Qwen3-VL-2B-Instruct** – Vision–language transformer used as the main architecture for caption generation.

---

# 📄 License

This project is created for educational purposes as part of the HSE curriculum.

