import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os

st.title("🚀 LLM Training Dashboard — Checkpoint Comparison Mode")

# ----------------- LOAD CHECKPOINTS LIST -----------------
checkpoints = sorted([f for f in os.listdir("checkpoints") if f.startswith("epoch")])

if len(checkpoints) < 2:
    st.warning("Need at least 2 checkpoints to compare. Train more epochs.")
    st.stop()

col1, col2 = st.columns(2)
choice_A = col1.selectbox("Model A", checkpoints)
choice_B = col2.selectbox("Model B", checkpoints, index=1)

modelA_path = f"checkpoints/{choice_A}"
modelB_path = f"checkpoints/{choice_B}"

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModelForCausalLM.from_pretrained(path).to(device)
    return tok, mod

tokenizerA, modelA = load_model(modelA_path)
tokenizerB, modelB = load_model(modelB_path)

st.success("Models Loaded Successfully")

# ================= PROMPT =======================
prompt = st.text_area("Enter prompt:", "The universe began with")

if st.button("Compare Outputs"):
    def generate(model, tok):
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=150,
                temperature=0.8,
                top_p=0.92,
                do_sample=True
            )
        return tok.decode(out[0], skip_special_tokens=True)

    outputA = generate(modelA, tokenizerA)
    outputB = generate(modelB, tokenizerB)

    col1, col2 = st.columns(2)
    col1.write(f"### 🅐 Output ({choice_A})")
    col1.write(outputA)

    col2.write(f"### 🅑 Output ({choice_B})")
    col2.write(outputB)
