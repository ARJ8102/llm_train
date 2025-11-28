from pathlib import Path

def save_checkpoint(model, tokenizer, out_dir, epoch):
    out_dir = Path(out_dir) / f'epoch-{epoch}'
    out_dir.mkdir(parents=True, exist_ok=True)

    m = model.module if hasattr(model, 'module') else model
    m.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'Checkpoint saved to {out_dir}')
