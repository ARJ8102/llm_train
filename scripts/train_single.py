"""
Single GPU training script for GPT-2 / TinyLlama
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.checkpoints import save_checkpoint
from utils.logging_utils import init_wandb
import wandb
from torch.cuda.amp import autocast, GradScaler


# ===================== 1. CONFIGS =====================
MODEL_NAME = "gpt2"              # later we load from configs/
DATASET_NAME = "wikitext"
SUBSET = "wikitext-2-raw-v1"
BLOCK_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 2
LR = 3e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ===================== 2. LOAD DATA =====================
dataset = load_dataset(DATASET_NAME, SUBSET)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no PAD token


# ===================== 3. TOKENIZE =====================
def tokenize(example):
    return tokenizer(example["text"])

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


# ===================== 4. CHUNK INTO FIXED LENGTH =====================
def chunk(example):
    ids = sum(example["input_ids"], [])
    masks = sum(example["attention_mask"], [])

    total = (len(ids) // BLOCK_SIZE) * BLOCK_SIZE

    return {
        "input_ids":       [ids[i:i+BLOCK_SIZE]   for i in range(0, total, BLOCK_SIZE)],
        "attention_mask":  [masks[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
    }


chunked = tokenized.map(chunk, batched=True)


# ===================== 5. COLLATE + DATALOADER =====================
def collate(batch):
    ids = [torch.tensor(x["input_ids"]) for x in batch]
    masks = [torch.tensor(x["attention_mask"]) for x in batch]

    ids = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    labels = ids.clone()
    return ids, masks, labels


loader = DataLoader(chunked["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)


# ===================== 6. LOAD MODEL =====================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR)

init_wandb("llm-training-system")


scaler = GradScaler()


# ===================== 7. TRAIN LOOP =====================
# ===================== 7. TRAIN LOOP =====================
for epoch in range(EPOCHS):
    total_loss = 0

    for step, batch in enumerate(loader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        if step > 200:
          break
        

        with autocast():
          output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
          loss = output.loss



        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        total_loss += loss.item()

        # 🔥 log to WandB every 50 steps
        if step % 50 == 0:
            print(f"[Epoch {epoch+1}] Step {step} Loss={loss.item():.4f}")
            wandb.log({"step_loss": loss.item()})

    # end-of-epoch summary
    avg = total_loss / len(loader)
    print(f"\nEpoch {epoch+1} completed  |  Avg Loss = {avg:.4f}\n")

    wandb.log({"epoch_loss": avg})

    # 💾 save model checkpoint each epoch
    save_checkpoint(model, tokenizer, "checkpoints", epoch+1)


