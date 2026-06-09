"""
DDP Training Script for GPT-2

Local test:
    torchrun --nproc_per_node=1 scripts/train_ddp.py

Multi-GPU:
    torchrun --nproc_per_node=4 scripts/train_ddp.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

from utils.checkpoints import save_checkpoint


MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 2
LR = 3e-5
MAX_STEPS_PER_EPOCH = 200


def setup_ddp():
    backend = "gloo" if os.name == "nt" else ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"[DDP] backend={backend}, world_size={world_size}, device={device}")

    return rank, world_size, local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


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
            "input_ids": [ids[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)],
            "attention_mask": [masks[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)],
        }

    return tokenized.map(chunk, batched=True)


def collate_fn(batch, pad_id):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    masks = [torch.tensor(x["attention_mask"]) for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    labels = input_ids.clone()
    return input_ids, masks, labels


def main():
    rank, world_size, local_rank, device = setup_ddp()

    if rank == 0:
        print("Loading tokenizer and dataset...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer)
    chunked = chunk_dataset(tokenized)
    train_ds = chunked["train"]

    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    if rank == 0:
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(EPOCHS):
        ddp_model.train()
        sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(loader):
            if step > MAX_STEPS_PER_EPOCH:
                break

            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with autocast(enabled=torch.cuda.is_available()):
                outputs = ddp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_steps += 1

            if step % 50 == 0 and rank == 0:
                print(f"[DDP Epoch {epoch + 1}] Step {step} Loss={loss.item():.4f}")

        avg_loss_tensor = torch.tensor(total_loss / max(1, num_steps), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / world_size).item()

        if rank == 0:
            print(f"DDP Epoch {epoch + 1} done | Avg Loss={avg_loss:.4f}")
            save_checkpoint(ddp_model.module, tokenizer, "checkpoints/ddp", epoch + 1)

    if rank == 0:
        print("DDP training complete.")

    cleanup_ddp()


if __name__ == "__main__":
    main()