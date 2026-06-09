"""
Evaluate a saved GPT-style checkpoint on WikiText-2 validation data.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/single/epoch-2
"""

import argparse
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 4
MAX_EVAL_STEPS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved checkpoint folder, e.g. checkpoints/single/epoch-2",
    )
    return parser.parse_args()


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
    args = parse_args()

    if not os.path.isdir(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint folder not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)
    model.eval()

    tokenized = tokenize_dataset(tokenizer)
    chunked = chunk_dataset(tokenized)
    eval_ds = chunked["validation"]

    loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= MAX_EVAL_STEPS:
                break

            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    perplexity = math.exp(avg_loss)

    print("\nEvaluation complete")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Eval loss  : {avg_loss:.4f}")
    print(f"Perplexity : {perplexity:.4f}")


if __name__ == "__main__":
    main()