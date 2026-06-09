import os

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(page_title="LLM Training Dashboard", layout="wide")

st.title("🚀 LLM Training Dashboard")
st.caption("Compare GPT-style checkpoints trained with Single GPU, DDP, FSDP, and ZeRO.")


CHECKPOINT_ROOT = "checkpoints"
STRATEGIES = ["single", "ddp", "fsdp", "zero"]


def discover_checkpoints():
    found = []

    if not os.path.isdir(CHECKPOINT_ROOT):
        return found

    for strategy in STRATEGIES:
        strategy_dir = os.path.join(CHECKPOINT_ROOT, strategy)

        if not os.path.isdir(strategy_dir):
            continue

        epochs = sorted(
            [
                name
                for name in os.listdir(strategy_dir)
                if name.startswith("epoch")
                and os.path.isdir(os.path.join(strategy_dir, name))
            ]
        )

        for epoch in epochs:
            checkpoint_path = os.path.join(strategy_dir, epoch)

            # Only include Hugging Face-compatible checkpoints
            if os.path.exists(os.path.join(checkpoint_path, "config.json")):
                found.append(
                    {
                        "label": f"{strategy.upper()} / {epoch}",
                        "strategy": strategy,
                        "epoch": epoch,
                        "path": checkpoint_path,
                    }
                )

    return found


checkpoints = discover_checkpoints()

if len(checkpoints) < 2:
    st.warning(
        "Need at least 2 Hugging Face-compatible checkpoints to compare. "
        "Expected folders like checkpoints/single/epoch-1 or checkpoints/ddp/epoch-2."
    )
    st.stop()


device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"**Device:** `{device}`")
st.sidebar.write(f"**Checkpoints found:** `{len(checkpoints)}`")

with st.sidebar.expander("Available checkpoints", expanded=True):
    for ckpt in checkpoints:
        st.write(f"- {ckpt['label']}")


labels = [ckpt["label"] for ckpt in checkpoints]
label_to_ckpt = {ckpt["label"]: ckpt for ckpt in checkpoints}


col1, col2 = st.columns(2)

choice_a_label = col1.selectbox("Model A", labels, index=0)
choice_b_label = col2.selectbox(
    "Model B",
    labels,
    index=1 if len(labels) > 1 else 0,
)

ckpt_a = label_to_ckpt[choice_a_label]
ckpt_b = label_to_ckpt[choice_b_label]


@st.cache_resource(show_spinner=True)
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()

    return tokenizer, model


with st.spinner("Loading selected checkpoints..."):
    tokenizer_a, model_a = load_model(ckpt_a["path"])
    tokenizer_b, model_b = load_model(ckpt_b["path"])

st.success("Models loaded successfully.")


st.subheader("Checkpoint details")

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.markdown("### Model A")
    st.write(f"**Strategy:** `{ckpt_a['strategy']}`")
    st.write(f"**Epoch:** `{ckpt_a['epoch']}`")
    st.write(f"**Path:** `{ckpt_a['path']}`")

with detail_col2:
    st.markdown("### Model B")
    st.write(f"**Strategy:** `{ckpt_b['strategy']}`")
    st.write(f"**Epoch:** `{ckpt_b['epoch']}`")
    st.write(f"**Path:** `{ckpt_b['path']}`")


st.subheader("Prompt comparison")

prompt = st.text_area(
    "Enter prompt",
    "The universe began with",
    height=120,
)

gen_col1, gen_col2, gen_col3 = st.columns(3)

max_new_tokens = gen_col1.slider("Max new tokens", 20, 150, 80, step=10)
temperature = gen_col2.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)
top_p = gen_col3.slider("Top-p", 0.1, 1.0, 0.92, step=0.01)


def generate(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


if st.button("Compare outputs"):
    with st.spinner("Generating outputs..."):
        output_a = generate(model_a, tokenizer_a, prompt)
        output_b = generate(model_b, tokenizer_b, prompt)

    out_col1, out_col2 = st.columns(2)

    with out_col1:
        st.markdown(f"### 🅐 {choice_a_label}")
        st.write(output_a)

    with out_col2:
        st.markdown(f"### 🅑 {choice_b_label}")
        st.write(output_b)