import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id):
    ids = [torch.tensor(x['input_ids']) for x in batch]
    ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    labels = ids.clone()
    return ids, labels
