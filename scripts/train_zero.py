"""
ZeRO Training Script using DeepSpeed.

Run on 2 GPUs:
    deepspeed --num_gpus=2 scripts/train_zero.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import deepspeed
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.checkpoints import save_checkpoint


MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 2
LR = 3e-5
MAX_STEPS_PER_EPOCH = 200


DS_CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "train_batch_size": 2,
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 2
    }
}


def tokenize_dataset(tokenizer):
    dataset = load_dataset(DATASET_NAME, SUBSET)

    def tokenize(examples):
        return tokenizer(examples["text"])

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def chunk_dataset(tokenized):
    def chunk(examples):
        ids = sum(examples["input_ids"], [])
        masks = sum(examples["attention_mask"], [])

        total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
        ids = ids[:total]
        masks = masks[:total]

        return {
            "input_ids": [
                ids[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)
            ],
            "attention_mask": [
                masks[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)
            ],
        }

    return tokenized.map(chunk, batched=True)


def collate_fn(batch, pad_id):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    masks = [torch.tensor(x["attention_mask"]) for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    labels = input_ids.clone()
    labels[labels == pad_id] = -100

    return input_ids, masks, labels


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer)
    chunked = chunk_dataset(tokenized)
    train_ds = chunked["train"]

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=DS_CONFIG,
    )

    for epoch in range(EPOCHS):
        model_engine.train()
        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(loader):
            if step > MAX_STEPS_PER_EPOCH:
                break

            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            num_steps += 1

            if step % 50 == 0 and model_engine.global_rank == 0:
                print(
                    f"[ZeRO Epoch {epoch + 1}] Step {step} "
                    f"Loss={loss.item():.4f}"
                )

        if model_engine.global_rank == 0:
            avg_loss = total_loss / max(1, num_steps)
            print(f"ZeRO Epoch {epoch + 1} done | Avg Loss={avg_loss:.4f}")

            save_checkpoint(
                model_engine.module,
                tokenizer,
                "checkpoints/zero",
                epoch + 1,
            )

    if model_engine.global_rank == 0:
        print("ZeRO training complete.")


if __name__ == "__main__":
    main()