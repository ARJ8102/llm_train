import wandb

def init_wandb(project='llm-training-system'):
    wandb.init(project=project)
