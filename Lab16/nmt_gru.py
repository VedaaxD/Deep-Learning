# ----------------------- IMPORTS & SETUP -----------------------
import spacy                      # tokenizer (English: en_core_web_sm); Hindi: spaCy blank model used
import pandas as pd               # reading CSVs and simple table ops
import torch                      # main PyTorch package
import torch.nn as nn             # neural network building blocks
import torch.optim as optim       # optimizers (Adam etc.)
from matplotlib import pyplot as plt  # plotting losses and BLEU
from torch.utils.data import Dataset, DataLoader  # dataset / dataloader abstractions
from sklearn.model_selection import train_test_split  # train/test splitting utility
from collections import Counter  # frequency counting for vocab building
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # BLEU evaluation
from tqdm import tqdm             # progress bars
import os                         # file system operations

# ----------------------- CONFIG / HYPERPARAMETERS -----------------------
# device: use GPU if available, else CPU. Always check this early to avoid silent CPU training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants / hyperparameters — tune based on GPU memory and dataset size
MAX_LEN = 12           # max tokens per sentence (smaller -> faster, but truncates long sentences)
BATCH_SIZE = 64        # batch size for training/eval
EMB_DIM = 128          # token embedding dimension
HIDDEN_DIM = 256       # GRU hidden size
EPOCHS = 100           # number of training epochs (increase if underfitting; watch time)
LR = 0.001             # learning rate for optimizer
MIN_FREQ = 1           # minimum token frequency to include in vocab (set >1 to reduce vocab)
NUM_WORKERS = 6        # DataLoader workers (increase on machines with many CPUs)
SAVE_DIR = "Hin2Eng_Model"  # where models and plots will be saved
os.makedirs(SAVE_DIR, exist_ok=True)  # create save directory if it doesn't exist

# ----------------------- DATASET CLASS -----------------------
class Hin2Eng(Dataset):
    """
    Basic PyTorch Dataset that stores pre-tokenized & encoded lists of equal-length sequences.
    Each item returned is: (source_tensor, target_tensor)
    """
    def __init__(self, source_list, target_list):
        # convert lists of ints to long tensors so they can be fed to Embedding layers
        self.source_tensor = torch.tensor(source_list, dtype=torch.long)
        self.target_tensor = torch.tensor(target_list, dtype=torch.long)

    def __len__(self):
        # number of samples
        return len(self.source_tensor)

    def __getitem__(self, idx):
        # return a single pair (src, trg)
        return self.source_tensor[idx], self.target_tensor[idx]

# ----------------------- DATA PREPROCESSING -----------------------
def Data_Preprocessing(data):
    """
    Inputs:
      - data: pandas DataFrame with 'english_sentence' and 'hindi_sentence' columns
    Outputs:
      - Train and Test DataLoaders, vocabularies (dicts), index->token maps (itos),
        and the spaCy tokenizers used.
    Notes:
      - This function lowercases, tokenizes, builds vocabulary, encodes & pads sequences.
      - If you change MAX_LEN, re-run this to re-encode sequences.
    """
    # special tokens (consistent indices matter across model and loss function)
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # ensure strings, lowercase and strip whitespace — consistent formatting reduces rare tokens
    data["english_sentence"] = data["english_sentence"].astype(str).str.lower().str.strip()
    data["hindi_sentence"] = data["hindi_sentence"].astype(str).str.lower().str.strip()

    # tokenizers:
    # - english uses spaCy pretrained small model (better tokenization)
    # - hindi uses spaCy blank model (simple rule-based tokenization); consider indic-nlp for better Hindi tokenization
    english = spacy.load("en_core_web_sm")
    hindi = spacy.blank("hi")

    # helper tokenizer wrapper: returns list[str]
    def tokenize(tokenizer, sentence):
        # tokenizer can be a spaCy model or blank; calling it returns tokens with .text
        return [token.text.lower() for token in tokenizer(sentence)]

    # build_vocab: count tokens and produce token->index and index->token maps
    def build_vocab(texts, tokenizer, min_freq=1):
        counter = Counter()
        for text in texts:
            # guard against NaN or non-string values
            if isinstance(text, str):
                tokens = tokenize(tokenizer, text)
                counter.update(tokens)
        # reserve indices for special tokens
        vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        # itos (index -> token) useful during decoding/evaluation
        itos = {i: tok for tok, i in vocab.items()}
        return vocab, itos

    # encode: converts sentence -> list of token indices with SOS and EOS and padding/truncation
    def encode(sentence, vocab, tokenizer, max_length=MAX_LEN):
        # convert words to indices (use UNK index if not found)
        tokens = tokenize(tokenizer, sentence)
        ids = [vocab[SOS_TOKEN]] + [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens] + [vocab[EOS_TOKEN]]
        # enforce length: truncate if too long, pad to max_length
        ids = ids[:max_length]
        while len(ids) < max_length:
            ids.append(vocab[PAD_TOKEN])
        return ids

    # prepare text lists
    hindi_texts = data["hindi_sentence"].tolist()
    english_texts = data["english_sentence"].tolist()

    # build vocabularies from the corpus texts
    source_vocab, source_itos = build_vocab(hindi_texts, hindi, MIN_FREQ)
    target_vocab, target_itos = build_vocab(english_texts, english, MIN_FREQ)

    # encode every sentence into fixed-length integer sequences
    source_sequences, target_sequences = [], []
    for _, row in data.iterrows():
        hi_ids = encode(row["hindi_sentence"], source_vocab, hindi)
        en_ids = encode(row["english_sentence"], target_vocab, english)
        source_sequences.append(hi_ids)
        target_sequences.append(en_ids)

    # split into train/test once encoded (random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(source_sequences, target_sequences, test_size=0.2, random_state=42)

    # wrap into Dataset and DataLoader (shuffle training data)
    Train_dataset = Hin2Eng(X_train, y_train)
    Test_dataset = Hin2Eng(X_test, y_test)
    Train_loader = DataLoader(Train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    Test_loader = DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"✅ Data Ready | Source vocab: {len(source_vocab)}, Target vocab: {len(target_vocab)}")
    # return everything you might need later
    return Train_loader, Test_loader, source_vocab, target_vocab, source_itos, target_itos, hindi, english

# ----------------------- MODEL COMPONENTS -----------------------
class Encoder(nn.Module):
    """
    Encoder uses a GRU RNN. It takes a batch of source token indices shaped (batch, seq_len),
    transposes them to (seq_len, batch), embeds tokens, runs GRU and returns the final hidden state.
    For GRU: hidden shape = (num_layers * num_directions, batch, hidden_dim)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        # embedding layer maps token indices -> embedding vectors for each token
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # single-layer unidirectional GRU; set batch_first=False because we transpose input
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src):
        # src: [batch, seq_len] -> transpose to [seq_len, batch] for nn.GRU with default setting
        src = src.transpose(0, 1)
        embedded = self.embedding(src)  # [seq_len, batch, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [seq_len, batch, hidden_dim], hidden: [1, batch, hidden_dim]
        return hidden  # return hidden state to initialize decoder

class Decoder(nn.Module):
    """
    Decoder also uses a GRU and returns logits for the next token.
    It receives a single token at each step (input) and previous hidden state.
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # map hidden -> vocab logits

    def forward(self, input, hidden):
        """
        input: [batch] (token indices for this timestep)
        hidden: [1, batch, hidden_dim]
        Returns:
          - prediction: [batch, vocab_size] (logits for next token)
          - hidden: updated hidden state
        """
        input = input.unsqueeze(0)  # make shape [1, batch] (seq_len=1)
        embedded = self.embedding(input)  # [1, batch, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [1, batch, hidden_dim]
        prediction = self.fc_out(output.squeeze(0))  # [batch, vocab_size]
        return prediction, hidden

class Seq2Seq(nn.Module):
    """
    A simple loop-based seq2seq (no attention). Encoder encodes source -> hidden,
    Decoder decodes step-by-step using teacher forcing optionally.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=MAX_LEN, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        trg: [batch, trg_len] or None (for inference)
        returns: outputs tensor shaped [trg_len, batch, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1] if trg is not None else max_len
        trg_vocab_size = self.decoder.fc_out.out_features

        # container to store predictions for each timestep
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encode source to get initial decoder hidden state
        hidden = self.encoder(src)  # for GRU: shape [1, batch, hidden_dim]

        # start token (SOS) index is 1 when we built vocab in preprocessing
        input = torch.ones(batch_size, dtype=torch.long).to(self.device)  # [batch]

        for t in range(trg_len):
            # pass current input token and hidden state to decoder
            output, hidden = self.decoder(input, hidden)  # output: [batch, vocab_size]
            outputs[t] = output  # store logits for timestep t

            # decide if we will use teacher forcing
            teacher_force = trg is not None and torch.rand(1).item() < teacher_forcing_ratio
            # if teacher forcing: next input = actual next token from trg, else = model's argmax
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

# ----------------------- TRAINING & EVALUATION HELPERS -----------------------
def train_epoch(model, dataloader, criterion, optimizer):
    """
    Runs one training epoch over dataloader. Returns average loss.
    Important: output shape from model is [trg_len, batch, vocab_size], while trg is [batch, trg_len].
    We reshape to compute CrossEntropyLoss across (batch * time) elements.
    """
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for src, trg in loop:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)  # [trg_len, batch, vocab_size]
        # flatten for loss: move time dim to front -> [batch, trg_len, vocab] -> flatten to 2D
        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        # backward & optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, target_itos):
    """
    Evaluate model: compute average loss and BLEU on dataloader.
    Returns (avg_loss, bleu, candidates, references)
    - candidates: list of predicted token-lists (words)
    - references: list of reference token-lists wrapped (for corpus_bleu)
    """
    model.eval()
    total_loss = 0
    candidates, references = [], []
    chencherry = SmoothingFunction()

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for src, trg in loop:
            src, trg = src.to(device), trg.to(device)
            # inference mode: teacher_forcing_ratio=0 ensures decoder uses its own predictions
            output = model(src, trg, teacher_forcing_ratio=0)  # [trg_len, batch, vocab_size]
            loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
            total_loss += loss.item()

            # convert logits -> predicted token ids: output.argmax over vocab dim
            preds = output.argmax(2).transpose(0, 1)  # shape -> [batch, trg_len]
            # iterate samples in batch to build token lists for BLEU
            for i in range(preds.shape[0]):
                pred_tokens, true_tokens = [], []
                # predicted tokens -> map ids to words until EOS (2) or we're done
                for idx in preds[i]:
                    token_id = idx.item()
                    if token_id == 2:   # EOS index = 2, stop
                        break
                    # ignore padding (0) and unknown mapping safety: ensure id exists in itos
                    if token_id != 0 and token_id in target_itos:
                        pred_tokens.append(target_itos[token_id])
                # same for target sequence
                for idx in trg[i]:
                    token_id = idx.item()
                    if token_id == 2:
                        break
                    if token_id != 0 and token_id in target_itos:
                        true_tokens.append(target_itos[token_id])
                # only add pair if both pred and true are non-empty (prevents BLEU errors)
                if pred_tokens and true_tokens:
                    candidates.append(pred_tokens)
                    references.append([true_tokens])
            loop.set_postfix(loss=loss.item())

    # compute corpus BLEU with smoothing (chencherry.method4)
    bleu = corpus_bleu(references, candidates, smoothing_function=chencherry.method4)
    return total_loss / len(dataloader), bleu, candidates, references

# ----------------------- MAIN TRAINING SCRIPT -----------------------
if __name__ == "__main__":
    # load data CSV (ensure correct path)
    data = pd.read_csv("Hindi_English_Truncated_Corpus.csv")

    # preprocess to get dataloaders, vocabularies and tokenizers
    Train_loader, Test_loader, src_vocab, tgt_vocab, src_itos, tgt_itos, hindi_tok, eng_tok = Data_Preprocessing(data)

    # instantiate encoder, decoder and seq2seq model and push to device
    encoder = Encoder(len(src_vocab), EMB_DIM, HIDDEN_DIM).to(device)
    decoder = Decoder(len(tgt_vocab), EMB_DIM, HIDDEN_DIM).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # optimizer and loss: CrossEntropy expects raw logits and integer class targets
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # ignore_index ensures we don't include PAD tokens in loss computation
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
    # learning rate scheduler: halves LR every 10 epochs to help finer convergence later
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # tracking best metrics and training history for plotting/saving
    best_val_loss = float("inf")
    best_bleu = 0
    train_history, val_history, bleu_history = [], [], []

    print("Starting training...\n")
    for epoch in range(EPOCHS):
        # train for one epoch
        train_loss = train_epoch(model, Train_loader, criterion, optimizer)
        # evaluate on validation/test set
        val_loss, bleu_score, translation, references = evaluate(model, Test_loader, criterion, tgt_itos)
        # scheduler step after validation
        scheduler.step()

        # save metrics
        train_history.append(train_loss)
        val_history.append(val_loss)
        bleu_history.append(bleu_score)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.4f}")

        # ----------------- model checkpointing -----------------
        # save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_val_loss_model.pth"))
            print(f"✅ Best Val Loss model saved at epoch {epoch+1}")

        # save best model by BLEU score
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_bleu_model.pth"))
            print(f"🌟 Best BLEU model saved at epoch {epoch+1} (BLEU: {bleu_score:.4f})")

        # periodic checkpoint every 50 epochs (saves intermediate progress)
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved at {ckpt_path}")

        # free GPU cache between epochs (useful if you see increasing memory)
        torch.cuda.empty_cache()

    print(f"\n✅ Training complete! Best Val Loss: {best_val_loss:.4f}, Best BLEU: {best_bleu:.4f}")

    # ----------------------- PLOTTING -----------------------
    # plot and save train/val loss figure
    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Val Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Training and Validation Loss")
    plt.legend(); plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))
    plt.show()

    # plot and save BLEU history
    plt.figure(figsize=(8, 5))
    plt.plot(bleu_history, label="BLEU Score", color='orange')
    plt.xlabel("Epochs"); plt.ylabel("BLEU"); plt.title("BLEU Score Over Time")
    plt.legend(); plt.savefig(os.path.join(SAVE_DIR, "bleu_plot.png"))
    plt.show()
