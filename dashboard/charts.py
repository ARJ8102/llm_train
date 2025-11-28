import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    return plt
