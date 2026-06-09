"""
FSDP Training Script for GPT-2

Best tested on Linux/HPC with GPUs:
    torchrun --nproc_per_node=2 scripts/train_fsdp.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from utils.checkpoints import save_checkpoint


MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 2
LR = 5e-6
MAX_STEPS_PER_EPOCH = 200


def setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
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
        print(f"[FSDP] backend={backend}, world_size={world_size}, device={device}")

    return rank, world_size, local_rank, device


def cleanup():
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
    labels[labels == pad_id] = -100
    return input_ids, masks, labels


def main():
    rank, world_size, local_rank, device = setup()

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

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    if torch.cuda.is_available():
        fsdp_model = FSDP(model, device_id=device)
    else:
        fsdp_model = FSDP(model)

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=LR)
    scaler = GradScaler(enabled=False)

    for epoch in range(EPOCHS):
        fsdp_model.train()
        sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0

        for step, batch in enumerate(loader):
            if step > MAX_STEPS_PER_EPOCH:
                break

            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            with autocast(enabled=False):
                outputs = fsdp_model(
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
                print(f"[FSDP Epoch {epoch + 1}] Step {step} Loss={loss.item():.4f}")

        avg_loss_tensor = torch.tensor(total_loss / max(1, num_steps), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / world_size).item()

        if rank == 0:
            print(f"FSDP Epoch {epoch + 1} done | Avg Loss={avg_loss:.4f}")

        dist.barrier()

        full_state_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True
        )

        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            full_state_config,
        ):
            state_dict = fsdp_model.state_dict()

        if rank == 0:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            base_model.load_state_dict(state_dict, strict=True)
            save_checkpoint(base_model, tokenizer, "checkpoints/fsdp", epoch + 1)
            print(f"FSDP checkpoint saved for epoch {epoch + 1}")

        dist.barrier()

    cleanup()


if __name__ == "__main__":
    main()