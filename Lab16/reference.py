# Sequence-to-Sequence Translation (English <-> Hindi) using PyTorch
# Implements both LSTM and GRU encoder-decoder architectures
# Dataset: https://www.kaggle.com/code/aiswaryaramachandran/english-to-hindi-neural-machine-translation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# -------------------------------
# Config
# -------------------------------
DATA_PATH = 'eng-hin.csv'  # Path to dataset
MAX_SAMPLES = 30000
BATCH_SIZE = 64
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Data Preprocessing
# -------------------------------
ENG_REGEX = re.compile(r"[^a-zA-Z0-9.,!?;'\- ]+")
HIN_REGEX = re.compile(r"[^\u0900-\u097F0-9a-zA-Z.,!?;'\- ]+")

def clean_sentence(sent, lang='eng'):
    sent = sent.strip()
    sent = re.sub(r"\s+", " ", sent)
    if lang == 'eng':
        sent = sent.lower()
        sent = ENG_REGEX.sub('', sent)
    else:
        sent = HIN_REGEX.sub('', sent)
    return sent

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
eng, hin = df.iloc[:, 0], df.iloc[:, 1]

pairs = []
for e, h in zip(eng, hin):
    if isinstance(e, str) and isinstance(h, str):
        e, h = clean_sentence(e, 'eng'), clean_sentence(h, 'hin')
        pairs.append((e, '<sos> ' + h + ' <eos>'))

if MAX_SAMPLES:
    pairs = pairs[:MAX_SAMPLES]

train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

# -------------------------------
# Tokenization
# -------------------------------
from collections import Counter

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(sent.split())
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(sentence, vocab):
    return [vocab.get(w, vocab['<unk>']) for w in sentence.split()]

def pad_sequences(sequences, pad_idx):
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_idx] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)

eng_vocab = build_vocab([p[0] for p in train_pairs])
hin_vocab = build_vocab([p[1] for p in train_pairs])
inv_hin_vocab = {v: k for k, v in hin_vocab.items()}

# -------------------------------
# Dataset Class
# -------------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = encode(src, self.src_vocab)
        trg_ids = encode(trg, self.trg_vocab)
        return torch.tensor(src_ids), torch.tensor(trg_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequences(src_batch, pad_idx=0)
    trg_padded = pad_sequences(trg_batch, pad_idx=0)
    return src_padded, trg_padded

train_dataset = TranslationDataset(train_pairs, eng_vocab, hin_vocab)
val_dataset = TranslationDataset(val_pairs, eng_vocab, hin_vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# -------------------------------
# Seq2Seq Models (LSTM and GRU)
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, model_type='lstm'):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.model_type = model_type
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, model_type='lstm'):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.model_type = model_type
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        if self.model_type == 'lstm':
            output, (h, c) = self.rnn(embedded, hidden)
            pred = self.fc(output.squeeze(1))
            return pred, (h, c)
        else:
            output, h = self.rnn(embedded, hidden)
            pred = self.fc(output.squeeze(1))
            return pred, h

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        hidden = self.encoder(src)
        if isinstance(hidden, tuple):
            input_token = trg[:, 0]
            for t in range(1, trg_len):
                output, hidden = self.decoder(input_token, hidden)
                outputs[:, t] = output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input_token = trg[:, t] if teacher_force else top1
        else:
            input_token = trg[:, 0]
            for t in range(1, trg_len):
                output, hidden = self.decoder(input_token, hidden)
                outputs[:, t] = output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input_token = trg[:, t] if teacher_force else top1

        return outputs

# -------------------------------
# Initialize Model (LSTM or GRU)
# -------------------------------
model_type = 'lstm'  # change to 'gru' to use GRU version

encoder = Encoder(len(eng_vocab), EMBED_SIZE, HIDDEN_SIZE, model_type)
decoder = Decoder(len(hin_vocab), EMBED_SIZE, HIDDEN_SIZE, model_type)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------
# Training Loop
# -------------------------------
print("Training started...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

print("Training complete.")
#_______________________________________________________________________________________________________________________
# English ↔ Hindi Neural Machine Translation using Seq2Seq (LSTM / GRU)
# Dataset: Hindi_English_Truncated_Corpus.csv
# Reference merged with your preprocessing steps
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Step 1: Load and clean dataset
# ------------------------------
print("Loading dataset...")
lines = pd.read_csv("Hindi_English_Truncated_Corpus.csv")

print("No. of sources present:")
print(lines["source"].value_counts())

# Drop NaNs and duplicates
lines = lines[~pd.isnull(lines["english_sentence"])]
lines.drop_duplicates(inplace=True)

# Cleaning English and Hindi sentences
lines["english_sentence"] = lines["english_sentence"].apply(lambda x: x.lower())
lines["english_sentence"] = lines["english_sentence"].apply(lambda x: re.sub(r"[^\w\s]", "", x))
lines["hindi_sentence"] = lines["hindi_sentence"].apply(lambda x: re.sub(r"[^\w\s]", "", x))
lines["english_sentence"] = lines["english_sentence"].apply(lambda x: re.sub(r"\d+", "", x))
lines["hindi_sentence"] = lines["hindi_sentence"].apply(lambda x: re.sub(r"\d+", "", x))

# Add start and end tokens to Hindi sentences
lines["hindi_sentence"] = lines["hindi_sentence"].apply(lambda x: "START_ " + x + " _END")

print("Sample cleaned data:")
print(lines.head(10))

# ------------------------------
# Step 2: Build Vocabulary
# ------------------------------
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word

def encode(sentence, vocab):
    return [vocab.get(w, vocab["<UNK>"]) for w in sentence.split()]

def pad_sequences(sequences, pad_idx):
    max_len = max(len(s) for s in sequences)
    return torch.tensor([s + [pad_idx]*(max_len - len(s)) for s in sequences], dtype=torch.long)

eng_vocab, eng_idx2word = build_vocab(lines["english_sentence"])
hin_vocab, hin_idx2word = build_vocab(lines["hindi_sentence"])
print(f"English vocab size: {len(eng_vocab)}, Hindi vocab size: {len(hin_vocab)}")

# ------------------------------
# Step 3: Train/Val Split
# ------------------------------
pairs = list(zip(lines["english_sentence"], lines["hindi_sentence"]))
train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

# ------------------------------
# Step 4: Dataset & DataLoader
# ------------------------------
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = encode(src, self.src_vocab)
        trg_ids = encode(trg, self.trg_vocab)
        return torch.tensor(src_ids), torch.tensor(trg_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequences(src_batch, pad_idx=0)
    trg_padded = pad_sequences(trg_batch, pad_idx=0)
    return src_padded, trg_padded

train_dataset = TranslationDataset(train_pairs, eng_vocab, hin_vocab)
val_dataset = TranslationDataset(val_pairs, eng_vocab, hin_vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

# ------------------------------
# Step 5: Seq2Seq Model (LSTM/GRU)
# ------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, model_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.model_type = model_type
        if model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.rnn(emb)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, model_type="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.model_type = model_type
        if model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        emb = self.embedding(x)
        if self.model_type == "lstm":
            output, (h, c) = self.rnn(emb, hidden)
            pred = self.fc(output.squeeze(1))
            return pred, (h, c)
        else:
            output, h = self.rnn(emb, hidden)
            pred = self.fc(output.squeeze(1))
            return pred, h

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        hidden = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

# ------------------------------
# Step 6: Initialize and Train
# ------------------------------
model_type = "lstm"  # or "gru"
input_dim = len(eng_vocab)
output_dim = len(hin_vocab)
embed_dim = 256
hidden_dim = 512

encoder = Encoder(input_dim, embed_dim, hidden_dim, model_type)
decoder = Decoder(output_dim, embed_dim, hidden_dim, model_type)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training started...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")

print("Training complete.")
