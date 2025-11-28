🚀 LLM Training System

Full training + evaluation pipeline for GPT-style models

Features
Module	Description
🔥 train_single.py	Train GPT-2/TinyLlama on one GPU w/ Mixed Precision
⚡ train_ddp.py	Multi-GPU Distributed Training (DDP, torchrun)
📊 Dashboard UI	Streamlit model comparison, side-by-side generation
💾 Checkpointing	Automatic save per epoch
🌍 WandB Logging	Live loss tracking + experiment history
1. Train on 1 GPU
python scripts/train_single.py

2. Distributed Multi-GPU
torchrun --nproc_per_node=4 scripts/train_ddp.py

3. Launch Dashboard
streamlit run dashboard/app.py


Then open UI — compare outputs of any two checkpoints.