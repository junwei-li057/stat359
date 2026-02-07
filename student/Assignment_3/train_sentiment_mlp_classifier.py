import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import gensim.downloader as api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Loading dataset")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree')
df = pd.DataFrame(dataset['train'])

# Load FastText
print("Loading FastText embeddings...")
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df['label'], random_state=SEED
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.15, stratify=train_val_df['label'], random_state=SEED
)

def get_mean_vector(text):
    words = text.lower().split()
    vectors = [fasttext_model[w] for w in words if w in fasttext_model]
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

print("Vectorizing sentences...")
X_train = np.array([get_mean_vector(t) for t in train_df['sentence']])
y_train = train_df['label'].values
X_val = np.array([get_mean_vector(t) for t in val_df['sentence']])
y_val = val_df['label'].values
X_test = np.array([get_mean_vector(t) for t in test_df['sentence']])
y_test = test_df['label'].values

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=64)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=64)

# Deal with imbalance
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(class_counts)
class_weights = torch.FloatTensor(weights).to(device)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, output_dim=3):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

model = MLPClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Iteration
epochs = 50
history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}
best_val_f1 = 0

if not os.path.exists('outputs'): os.makedirs('outputs')

print("Starting training...")
for epoch in range(epochs):
    model.train()
    train_loss, train_preds, train_true = 0, [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(outputs.argmax(1).cpu().numpy())
        train_true.extend(yb.cpu().numpy())
    
    # Validation
    model.eval()
    val_loss, val_preds, val_true = 0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            val_loss += criterion(outputs, yb).item()
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_true.extend(yb.cpu().numpy())
    
    t_f1 = f1_score(train_true, train_preds, average='macro')
    v_f1 = f1_score(val_true, val_preds, average='macro')
    t_acc = accuracy_score(train_true, train_preds)
    v_acc = accuracy_score(val_true, val_preds)
    
    history['train_loss'].append(train_loss/len(train_loader))
    history['val_loss'].append(val_loss/len(val_loader))
    history['train_f1'].append(t_f1)
    history['val_f1'].append(v_f1)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)
    
    print(f"Epoch {epoch+1:02d}: Val F1 = {v_f1:.4f}, Val Acc = {v_acc:.4f}")
    
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        torch.save(model.state_dict(), 'best_mlp_model.pth')
    
    scheduler.step(v_f1)

# Test & Plot
model.load_state_dict(torch.load('best_mlp_model.pth'))
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb.to(device))
        test_preds.extend(outputs.argmax(1).cpu().numpy())
        test_true.extend(yb.cpu().numpy())

print(f"\nFinal Test Macro F1: {f1_score(test_true, test_preds, average='macro'):.4f}")

def plot_metrics(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(['loss', 'acc', 'f1']):
        axes[i].plot(history[f'train_{metric}'], label='Train')
        axes[i].plot(history[f'val_{metric}'], label='Val')
        axes[i].set_title(f'MLP {metric.capitalize()}')
        axes[i].legend()
    plt.savefig('outputs/mlp_training_curves.png')

    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title('MLP Confusion Matrix')
    plt.savefig('outputs/mlp_confusion_matrix.png')

plot_metrics(history)