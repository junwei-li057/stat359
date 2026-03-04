import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_training_results(log_dir):

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    train_loss = pd.DataFrame(ea.Scalars('Loss/train'))
    val_loss = pd.DataFrame(ea.Scalars('Loss/val'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot：Loss
    ax1.plot(train_loss['step'], train_loss['value'], label='Train Loss', color='blue', alpha=0.6)
    ax1.plot(val_loss['step'], val_loss['value'], label='Val Loss', color='red', linewidth=2)
    ax1.set_title('Cross-Entropy Loss')
    ax1.set_xlabel('Steps')
    ax1.legend()

    # Right plot：Perplexity
    train_ppl = pd.DataFrame(ea.Scalars('Perplexity/train'))
    val_ppl = pd.DataFrame(ea.Scalars('Perplexity/val'))
    
    ax2.plot(train_ppl['step'], train_ppl['value'], label='Train PPL', color='blue', alpha=0.6)
    ax2.plot(val_ppl['step'], val_ppl['value'], label='Val PPL', color='red', linewidth=2)
    ax2.set_title('Perplexity ($e^{Loss}$)')
    ax2.set_xlabel('Steps')
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_training_results('tinystories_model')
plot_training_results('tinystories_lora')