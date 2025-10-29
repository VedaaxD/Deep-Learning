# ===============================================
# 🔹 IMPORT ALL REQUIRED LIBRARIES
# ===============================================
import pandas as pd                  # For loading and manipulating CSV data
import torch                         # For tensors and general deep learning
import torch.nn as nn                # For neural network building blocks (layers, loss)
from torch.utils.data import Dataset, DataLoader, random_split  # For dataset & batching
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # For BLEU score evaluation

# Select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================
# 1️⃣ DEFINE CUSTOM DATASET CLASS
# ===============================================
class TranslationDataset(Dataset):
    def __init__(self, csv_file, src_col='hindi_sentence', trg_col='english_sentence', max_len=20):
        # Load CSV data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        self.src_col = src_col
        self.trg_col = trg_col
        self.max_len = max_len

        # Build vocabularies for source (Hindi) and target (English)
        self.src_vocab = self.build_vocab(self.data[src_col])
        self.trg_vocab = self.build_vocab(self.data[trg_col])

        # Create word → index and index → word mappings for both languages
        self.src2idx = {w: i for i, w in enumerate(self.src_vocab)}  # word to id
        self.trg2idx = {w: i for i, w in enumerate(self.trg_vocab)}
        self.idx2src = {i: w for w, i in self.src2idx.items()}        # id to word
        self.idx2trg = {i: w for w, i in self.trg2idx.items()}

    # ------------------------------------------------
    # Helper function: build a vocabulary from sentences
    # ------------------------------------------------
    def build_vocab(self, sentences, min_freq=1):
        from collections import Counter
        counter = Counter()

        # Count how many times each word appears
        for sent in sentences:
            if isinstance(sent, str):
                counter.update(sent.lower().split())  # tokenizes and counts words
            elif pd.notna(sent):
                counter.update(str(sent).lower().split())

        # Add special tokens for padding, sequence start/end, and unknown words
        vocab = ["<PAD>", "<START>", "<END>", "<UNK>"]

        # Add words that occur at least min_freq times
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab.append(word)
        return vocab

    # ------------------------------------------------
    # Convert a sentence into list of token IDs
    # ------------------------------------------------
    def encode_sentence(self, sentence, vocab_map, max_len, add_start_end=False):
        # Convert each word to ID, or use <UNK> (unknown) if word not in vocab
        tokens = [vocab_map.get(w, vocab_map["<UNK>"]) for w in sentence.lower().split()]

        # Optionally add <START> and <END> tokens (for target sentences)
        if add_start_end:
            tokens = [vocab_map["<START>"]] + tokens + [vocab_map["<END>"]]

        # Pad short sequences or truncate long ones to match max_len
        if len(tokens) < max_len:
            tokens += [vocab_map["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return tokens

    # Returns total number of sentence pairs in dataset
    def __len__(self):
        return len(self.data)

    # Fetch a single (Hindi, English) pair, convert to tensor
    def __getitem__(self, idx):
        src_sent = self.data.iloc[idx][self.src_col]   # Hindi sentence
        trg_sent = self.data.iloc[idx][self.trg_col]   # English sentence

        # Encode Hindi sentence (no <START>/<END>)
        src_tokens = self.encode_sentence(src_sent, self.src2idx, self.max_len)

        # Encode English sentence (add <START>/<END>)
        trg_tokens = self.encode_sentence(trg_sent, self.trg2idx, self.max_len, add_start_end=True)

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(trg_tokens, dtype=torch.long)

# ===============================================
# 2️⃣ POSITIONAL ENCODING LAYER
# ===============================================
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()

        # Initialize a tensor (max_len × emb_dim) to hold sine/cosine values
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term used for sinusoidal variation
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))

        # Fill even indices with sine and odd indices with cosine
        pe[:, 0::2] = torch.sin(position * div_term)  # even columns
        pe[:, 1::2] = torch.cos(position * div_term)  # odd columns

        # Add batch dimension so it can be broadcast during addition
        pe = pe.unsqueeze(0)

        # Register as buffer (not a learnable parameter but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to embeddings
        return x + self.pe[:, :x.size(1), :]

# ===============================================
# 3️⃣ TRANSFORMER-BASED SEQ2SEQ MODEL
# ===============================================
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim=128, nhead=8, num_layers=2, max_len=20, trg_pad_idx=0):
        super().__init__()

        # Token embedding layers for source (Hindi) and target (English)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, emb_dim)

        # Positional encoding module (adds info about word order)
        self.positional_encoding = PositionalEncoding(emb_dim, max_len)

        # PyTorch Transformer (handles both encoder and decoder)
        self.transformer = nn.Transformer(
            d_model=emb_dim,              # size of embedding vectors
            nhead=nhead,                  # number of attention heads
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,          # hidden dimension of feedforward sublayer
            batch_first=True              # input shape: (batch, seq, features)
        )

        # Final linear layer to map decoder output → vocabulary logits
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)

        # Save padding index for masking
        self.trg_pad_idx = trg_pad_idx

    # Create mask for <PAD> tokens in source sentences
    def make_src_mask(self, src):
        return (src != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

    # Create target mask so model doesn't "peek" at future words
    def make_trg_mask(self, trg):
        trg_len = trg.size(1)
        mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
        return mask

    def forward(self, src, trg):
        # Convert word indices → embeddings
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))

        # Create target mask (causal mask)
        trg_mask = self.make_trg_mask(trg)

        # Pass through Transformer network
        output = self.transformer(src_emb, trg_emb, tgt_mask=trg_mask)

        # Final prediction: map to vocab logits
        output = self.fc_out(output)
        return output

# ===============================================
# 4️⃣ BLEU SCORE HELPERS (for evaluation)
# ===============================================
def ids_to_sentence(ids, idx2word):
    # Converts list of IDs → words, ignoring special tokens
    return [idx2word[i] for i in ids if i not in (0, 1, 2)]

def compute_bleu(preds, refs, idx2word):
    # Convert predicted & reference IDs → words for BLEU
    pred_sentences = [ids_to_sentence(p, idx2word) for p in preds]
    ref_sentences = [[ids_to_sentence(r, idx2word)] for r in refs]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(ref_sentences, pred_sentences, smoothing_function=smoothie)

# ===============================================
# 5️⃣ MAIN TRAINING FUNCTION
# ===============================================
def main():
    csv_file = "Hindi_English_Truncated_Corpus.csv"
    max_len = 20

    # Load dataset and split into train/test sets (80/20)
    dataset = TranslationDataset(csv_file, max_len=max_len)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Prepare DataLoaders for batching
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # Initialize Transformer model
    model = TransformerSeq2Seq(
        len(dataset.src_vocab),         # Hindi vocab size
        len(dataset.trg_vocab),         # English vocab size
        emb_dim=128,                    # Embedding size
        max_len=max_len,
        trg_pad_idx=dataset.trg2idx["<PAD>"]
    ).to(device)

    # Define optimizer and loss (ignore PAD tokens)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg2idx["<PAD>"])

    EPOCHS = 10
    for epoch in range(EPOCHS):
        # ---------------- TRAINING ----------------
        model.train()
        epoch_loss = 0

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(src, trg)
            output_dim = output.shape[-1]

            # Calculate loss (skip the <START> token by slicing [:,1:])
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f}")

        # ---------------- EVALUATION ----------------
        model.eval()
        all_preds, all_refs = [], []

        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg)

                # Get predicted IDs (argmax over vocabulary)
                for i in range(src.size(0)):
                    pred_ids = output[i].argmax(1).tolist()
                    all_preds.append(pred_ids)
                    all_refs.append(trg[i].tolist())

        # Compute BLEU score
        bleu_score = compute_bleu(all_preds, all_refs, dataset.idx2trg)
        print(f"Epoch {epoch+1} | BLEU-4: {bleu_score:.4f}")

# Run training
if __name__ == "__main__":
    main()
