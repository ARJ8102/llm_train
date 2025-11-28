"""
DDP Training Script for GPT-2

Run (Colab, 1 GPU test):
    torchrun --nproc_per_node=1 scripts/train_ddp.py

Later (4 GPUs):
    torchrun --nproc_per_node=4 scripts/train_ddp.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

# make sure "utils" is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.checkpoints import save_checkpoint


# ================== CONFIG ==================
MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 2
LR = 3e-5

USE_SMALL_SUBSET = True       # to keep runs fast
SMALL_TRAIN_SIZE = 8000       # number of examples to use


# ============== DDP SETUP / TEARDOWN ==========
def setup_ddp():
    """
    Initialize process group and return:
        rank, world_size, local_rank, device
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[DDP] world_size={world_size}, device={device}")

    return rank, world_size, local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


# ============== DATA PIPELINE (same idea as single GPU) ==========
def tokenize_dataset(tokenizer):
    dataset = load_dataset(DATASET_NAME, SUBSET)

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def chunk_dataset(tokenized):
    def chunk(examples):
        ids = sum(examples["input_ids"], [])
        masks = sum(examples["attention_mask"], [])

        total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE
        ids = ids[:total]
        masks = masks[:total]

        return {
            "input_ids": [ids[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)],
            "attention_mask": [masks[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)],
        }

    return tokenized.map(chunk, batched=True)


def collate_fn(batch, pad_id):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    masks = [torch.tensor(x["attention_mask"]) for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    labels = input_ids.clone()
    return input_ids, masks, labels


# ===================== MAIN TRAIN =====================
def main():
    rank, world_size, local_rank, device = setup_ddp()

    # only rank 0 prints a bit more
    if rank == 0:
        print("Loading tokenizer & dataset...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer)
    chunked = chunk_dataset(tokenized)

    train_ds = chunked["train"]
    if USE_SMALL_SUBSET:
        # keep it fast while testing
        train_ds = train_ds.select(range(min(SMALL_TRAIN_SIZE, len(train_ds))))

    # each process gets its own sampler over the dataset
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    if rank == 0:
        print("Building model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=LR)
    scaler = GradScaler()

    # ================= TRAIN LOOP ==================
    for epoch in range(EPOCHS):

     
        ddp_model.train()
        # sampler needs the epoch for proper shuffling across replicas
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(train_loader):


          
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            if step > 200:
              break

            with autocast():
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
                print(f"[Epoch {epoch+1}] Step {step} | Loss={loss.item():.4f}")

        # Average loss across processes for reporting
        avg_loss_tensor = torch.tensor(total_loss / max(1, num_steps), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / world_size).item()

        if rank == 0:
            print(f"Epoch {epoch+1} done | Average Loss (across GPUs) = {avg_loss:.4f}")

            # save checkpoint only on rank 0
            save_checkpoint(ddp_model, tokenizer, "checkpoints_ddp", epoch + 1)

    if rank == 0:
        print("Training complete. Cleaning up.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
