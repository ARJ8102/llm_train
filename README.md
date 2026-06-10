# Distributed LLM Training and Checkpoint Comparison Framework

A production-style framework for training, evaluating, and comparing GPT-style language models using multiple distributed training strategies.

## Overview

This project demonstrates how modern Large Language Models (LLMs) can be trained using different parallelization techniques and then evaluated through an interactive dashboard.

The framework supports:

* Single GPU Training
* Distributed Data Parallel (DDP)
* Fully Sharded Data Parallel (FSDP)
* DeepSpeed ZeRO Optimization
* Hugging Face Checkpoint Management
* Interactive Streamlit Dashboard
* Side-by-Side Model Comparison

The goal is to understand the trade-offs between different distributed training approaches and visualize how checkpoints evolve throughout training.

---

## Features

### Training Strategies

| Strategy   | Description                                       |
| ---------- | ------------------------------------------------- |
| Single GPU | Standard training on a single GPU                 |
| DDP        | Data parallel training across multiple GPUs       |
| FSDP       | Fully Sharded Data Parallel memory optimization   |
| ZeRO       | DeepSpeed ZeRO optimizer for large-scale training |

### Model Evaluation

* Text generation comparison
* Perplexity evaluation
* Training loss tracking
* Checkpoint benchmarking

### Interactive Dashboard

Users can:

* Select two checkpoints
* Compare generated text
* Compare training approaches
* Analyze model outputs
* Visualize training metrics

---

## Project Architecture

```text
Palmetto HPC Cluster
        │
        ▼
Distributed Training
(Single / DDP / FSDP / ZeRO)
        │
        ▼
Checkpoint Generation
        │
        ▼
Hugging Face Model Repository
        │
        ▼
Streamlit Dashboard
        │
        ▼
Interactive Comparison Interface
```

---

## Technologies Used

### Machine Learning

* PyTorch
* Transformers
* DeepSpeed
* FSDP
* Distributed Data Parallel (DDP)

### Deployment

* Hugging Face Hub
* Hugging Face Spaces
* Streamlit

### Development

* Python
* Git
* GitHub

---

## Training Infrastructure

Models were trained on Clemson University's Palmetto High Performance Computing Cluster.

Training experiments included:

* Single GPU Training
* Multi-GPU DDP Training
* FSDP Memory Sharding
* ZeRO Optimization

---

## Live Demo

Interactive Dashboard:

[Hugging Face Space Link]

Model Checkpoints:

[Hugging Face Checkpoint Repository]

---

## Results

The framework successfully:

* Trained GPT-style models using four distributed strategies
* Saved checkpoints in Hugging Face format
* Uploaded checkpoints to the Hugging Face Hub
* Enabled real-time checkpoint comparison through a web dashboard

This project demonstrates practical MLOps workflows for distributed LLM training and deployment.

---

## Future Improvements

* Add larger transformer architectures
* Support LoRA fine-tuning
* Add training metric visualizations
* Compare inference latency
* Add benchmark datasets
* Enable checkpoint-to-checkpoint performance tracking

---

## Author

Atharva Jadhav

M.S. Computer Science
Clemson University

Interested in Machine Learning, LLM Systems, Distributed Training, and MLOps.
