import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import gensim.downloader as api
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Loading dataset")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree')
df = pd.DataFrame(dataset['train'])

print("Loading FastText model...")
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df['label'], random_state=SEED
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.15, stratify=train_val_df['label'], random_state=SEED
)

def prepare_sequence(text, max_len=32):
    words = text.lower().split()
    vectors = [fasttext_model[w] for w in words if w in fasttext_model]
    
    if len(vectors) >= max_len:
        return np.array(vectors[:max_len])
    else:
        padding = [np.zeros(300) for _ in range(max_len - len(vectors))]
        return np.array(vectors + padding)

X_train = np.array([prepare_sequence(s) for s in train_df['sentence']])
X_val = np.array([prepare_sequence(s) for s in val_df['sentence']])
X_test = np.array([prepare_sequence(s) for s in test_df['sentence']])

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=64)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=64)

class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor(1.0 / class_counts).to(device)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=3, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(last_hidden)

model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Iteration
EPOCHS = 60
MIN_EPOCHS = 30 
PATIENCE = 7    # Add early stopping
history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}

best_val_f1 = 0
early_stop_counter = 0

if not os.path.exists('outputs'): os.makedirs('outputs')

print(f"Starting training:")
for epoch in range(EPOCHS):
    model.train()
    t_loss, t_true, t_preds = 0, [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item()
        t_preds.extend(outputs.argmax(1).cpu().numpy())
        t_true.extend(yb.cpu().numpy())
    
    model.eval()
    v_loss, v_true, v_preds = 0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            v_loss += criterion(outputs, yb).item()
            v_preds.extend(outputs.argmax(1).cpu().numpy())
            v_true.extend(yb.cpu().numpy())
    
    train_f1 = f1_score(t_true, t_preds, average='macro')
    val_f1 = f1_score(v_true, v_preds, average='macro')
    
    history['train_loss'].append(t_loss/len(train_loader))
    history['val_loss'].append(v_loss/len(val_loader))
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)
    history['train_acc'].append(accuracy_score(t_true, t_preds))
    history['val_acc'].append(accuracy_score(v_true, v_preds))

    print(f"Epoch {epoch+1:02d}: Val F1 = {val_f1:.4f} | Val Acc = {history['val_acc'][-1]:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
    
    if early_stop_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()
test_true, test_preds = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb.to(device))
        test_preds.extend(outputs.argmax(1).cpu().numpy())
        test_true.extend(yb.cpu().numpy())

print(f"\nFinal Test Macro F1 Score: {f1_score(test_true, test_preds, average='macro'):.4f}")

# Plots
def plot_results(history, test_true, test_preds):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))
    
    metrics = ['loss', 'acc', 'f1']
    for i, m in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.plot(epochs_range, history[f'train_{m}'], label='Train')
        plt.plot(epochs_range, history[f'val_{m}'], label='Val')
        plt.title(f'LSTM {m.capitalize()} vs. Epochs')
        plt.legend()
    plt.savefig('outputs/lstm_training_curves.png')

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_true, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title('LSTM Confusion Matrix')
    plt.savefig('outputs/lstm_confusion_matrix.png')

plot_results(history, test_true, test_preds)