import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context):
        v_c = self.in_embed(center)
        v_o = self.out_embed(context)

        if v_o.dim() == 3:
            v_c = v_c.unsqueeze(1)
            score = torch.sum(v_c * v_o, dim=2)
        else:
            score = torch.sum(v_c * v_o, dim=1)
        return score

    def get_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


# Load processed data
with open('processed_data.pkl', 'rb') as pd:
    data = pickle.load(pd)

pairs = list(zip(data['skipgram_df']['center'], data['skipgram_df']['context']))
word2idx = data['word2idx']
idx2word = data['idx2word']
vocab_size = len(word2idx)

# Precompute negative sampling distribution below
word_counts = np.zeros(vocab_size)
for center, context in pairs:
    word_counts[center] += 1
    word_counts[context] += 1

unigram_probs = word_counts ** 0.75 # 3/4 smoothing
unigram_probs = unigram_probs / unigram_probs.sum()
unigram_probs = torch.tensor(unigram_probs, dtype=torch.float)


# Device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dataset and DataLoader
dataset = SkipGramDataset(pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def make_targets(center, context, unigram_probs, n_neg_samples, device):
    batch_size = center.size(0)

    pos_targets = torch.ones(batch_size, dtype=torch.float, device=device)

    neg_context = torch.multinomial(
        unigram_probs,
        batch_size * n_neg_samples,
        replacement=True
    ).view(batch_size, n_neg_samples).to(device)

    mask = neg_context == context.unsqueeze(1)
    while mask.any():
        neg_context[mask] = torch.multinomial(
            unigram_probs,
            mask.sum().item(),
            replacement=True
        ).to(device)
        mask = neg_context == context.unsqueeze(1)

    neg_targets = torch.zeros(
        batch_size, n_neg_samples, dtype=torch.float, device=device
    )

    return pos_targets, neg_targets, neg_context


# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in tqdm(dataloader):
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad()

        pos_targets, neg_targets, neg_context = make_targets(
            center,
            context,
            unigram_probs,
            NEGATIVE_SAMPLES,
            device
        )

        pos_score = model(center, context)
        pos_loss = criterion(pos_score, pos_targets)

        neg_score = model(center, neg_context)
        neg_score = neg_score.view(-1)
        neg_targets = neg_targets.view(-1)
        neg_loss = criterion(neg_score, neg_targets)

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
