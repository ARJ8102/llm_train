from datasets import load_dataset

def load_text_dataset(name='wikitext', subset='wikitext-2-raw-v1'):
    return load_dataset(name, subset)
