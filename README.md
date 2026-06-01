# GPT From Scratch

A minimal but production-inspired implementation of a **decoder-only Transformer language model** built entirely in PyTorch.
The goal of this project is to deeply understand modern LLM architecture by implementing key components from scratch, without relying on high-level frameworks like HuggingFace Trainer.

---

##  Key Features
- Decoder-only Transformer architecture (GPT-style)
- Grouped Query Attention (GQA) for efficient inference
- Rotary Position Embeddings (RoPE)
- RMSNorm (pre-norm architecture)
- SwiGLU feed-forward network
- KV Cache for fast autoregressive decoding
- Top-k / Top-p sampling
- Repetition penalty for improved generation quality
- Custom training + inference pipeline

---

##  Architecture Overview
- Tokenizer → Embedding layer
- Stacked Transformer blocks
- Each block:
  - RMSNorm
  - GQA Attention + RoPE
  - SwiGLU MLP
- Output projection → vocabulary logits
- 
---
## 📁 Project Structure
```
Model/                  # Core model implementation
├── model.py           # Main GPT model
├── transformer_block.py
├── attention.py       # GQA + TransformersBlock 
├── utils.py           # RoPE + SwiGLU + RMSNorm 

Experiments/           # experiments & logs
Assets/                # Loss curves, LR schedule

Trainer.py             # Training loop
TrainingExample.ipynb  # End-to-end training + inference demo
```
---

##  Training
Training is implemented in:
`TrainingExample.ipynb`
This NoteBook demonstrates:
- Data preprocessing and tokenization
- Model initialization
- Full training loop using `trainer.py`
- Example inference
---

##  Results
The model demonstrates stable convergence during training. 
**Best Loss**: 2.21866

<p align="center">
  <img src="Assets/LossCurves.png" width="900"/>
</p>
<p align="center">
  <img src="Assets/LossCurvesLR.png" width="900"/>
</p>

---

##  Inference

Supports:
- Autoregressive text generation
- KV-cache optimized decoding
- Sampling strategies (Top-k / Top-p)
- Repetition penalty control
---

##  Purpose
This project was built for:
- Understanding Transformer internals
- Learning LLM training pipeline from scratch
- Experimenting with modern architectural improvements (GQA, RoPE, RMSNorm,SwiGLU)
---
##  Tech Stack
- PyTorch
- Python

---
##  Notes
This is an educational implementation, but follows modern LLM design patterns used in production-scale models.

