# IMAGE CAPTIONING PROBLEM using RNNs for the Flickr 8k dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

# -----------------------------
# Load image embeddings
# -----------------------------
image_dict = torch.load("image_embedding.pt")
print(f"Loaded Image Embeddings from {len(image_dict)} images.")
images = list(image_dict.values())
image_names = list(image_dict.keys())
print(type(images), images[0].shape)  # <class 'list'> torch.Size([512])

# -----------------------------
# Load captions
# -----------------------------
with open("captions.txt", "r", encoding="utf-8") as f:
    captions = f.readlines()
print(f"Loaded {len(captions)} captions.")

processed_captions = []
for line in captions:
    if ',' in line:
        _, caption_text = line.strip().split(',', 1)
    else:
        caption_text = line.strip()
    processed_captions.append(caption_text.lower())

# -----------------------------
# Build vocabulary
# -----------------------------
tokens = [word.lower() for line in processed_captions for word in line.strip().split()]
word_counts = Counter(tokens)
special_tokens = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]
vocab = special_tokens + list(word_counts.keys())
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")
print(f"Vocab:{vocab}")
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# -----------------------------
# Load GloVe embeddings
# -----------------------------
embed_dim = 50
embedding_matrix = torch.randn(vocab_size, embed_dim)

glove_path = "/home/ibab/Downloads/glove.2024.wikigiga.50d/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        if len(values) != embed_dim + 1:
            continue
        try:
            vector = np.array(values[1:], dtype=np.float32)
        except ValueError:
            continue
        if word in word_to_index:
            embedding_matrix[word_to_index[word]] = torch.from_numpy(vector)

# -----------------------------
# Convert captions to indices
# -----------------------------
def captions_to_indices(captions, word_to_index, max_len=20):
    indexed_captions = []
    for cap in captions:
        words = cap.strip().split()
        indices = [word_to_index["<SOS>"]] + [word_to_index.get(w, word_to_index["<UNK>"]) for w in words] + [word_to_index["<EOS>"]]
        if len(indices) < max_len:
            indices += [word_to_index["<PAD>"]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        indexed_captions.append(indices)
    return torch.tensor(indexed_captions, dtype=torch.long)

caption_indices = captions_to_indices(processed_captions, word_to_index)

# -----------------------------
# Dataset class for batching
# -----------------------------
class CaptionDataset(Dataset):
    def __init__(self, images, captions, image_names):
        self.images = images
        self.captions = captions
        self.image_names = image_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].float(), self.captions[idx].long(), self.image_names[idx]
dataset = CaptionDataset(images, caption_indices, image_names)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# RNN Model
# -----------------------------
class CaptionRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim):
        super(CaptionRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = True

        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.img_fc = nn.Linear(512, hidden_dim)

    def forward(self, img_features,max_len=20):
        batch_size = img_features.size(0)
        hidden = self.img_fc(img_features).unsqueeze(0)  # (1, batch_size, hidden_dim)
        inputs = torch.full((batch_size, 1), word_to_index["<SOS>"], dtype=torch.long, device=img_features.device)
        logits_seq = []

        for _ in range(max_len):
            embedded = self.embedding(inputs)
            out, hidden = self.rnn(embedded, hidden)
            logits = self.fc(out.squeeze(1))
            logits_seq.append(logits.unsqueeze(1))
            inputs = logits.argmax(1).unsqueeze(1)  # no teacher forcing

        return torch.cat(logits_seq, dim=1)  # (batch_size, max_len, vocab_size)

hidden_dim = 256
model = CaptionRNN(vocab_size=vocab_size, hidden_dim=hidden_dim, embed_dim=embed_dim)
print("CaptionRNN model created.")

# -----------------------------
# Training
# -----------------------------
criterion = nn.CrossEntropyLoss(ignore_index=word_to_index["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_images, batch_captions,_ in dataloader:
        optimizer.zero_grad()
        logits = model(batch_images, max_len=20)  # (batch_size, max_len, vocab_size)
        loss = criterion(logits.view(-1, vocab_size), batch_captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")

# -----------------------------
# Test on one sample
# -----------------------------
model.eval()
import random

with torch.no_grad():
    idx = random.randint(0, len(dataset) - 1)
    sample_image, _, sample_name = dataset[idx]  # get image, caption (ignored), and name
    sample_image = sample_image.unsqueeze(0)  # add batch dim
    logits = model(sample_image, max_len=20)
    prediction = logits.argmax(2).squeeze(0).tolist()

    caption = []
    for idx in prediction:
        word = index_to_word[idx]
        if word == "<EOS>":
            break
        if word != "<SOS>":
            caption.append(word)

    print(f"Image: {sample_name}")
    print(f"Caption: {' '.join(caption)}")

# performs a single forward pass of an image captioning model without teacher forcing, also known as an inference or generation pass.
#
# Breakdown of Your forward Method
# Your forward method is the core of the generation process. Here's what it does step-by-step:
#
# Initialize the Hidden State: It takes the 512-D image embedding, passes it through a linear layer (self.img_fc) to match the RNN's hidden dimension, and uses this as the initial hidden state (h₀). This is how the model gets the "topic" of the image.
#
# Provide the Start Signal: It creates a starting input tensor that contains the index for the <SOS> (Start of Sequence) token.
#
# Generate Word by Word (Autoregressively): It enters a loop to generate the caption. In each iteration:
#
# It takes the current input (starting with <SOS>).
#
# It gets the word embedding for that input.
#
# It feeds the embedding and the current hidden state into the RNN to get an output and a new hidden state.
#
# It passes the RNN's output through a final linear layer (self.fc) to get scores (logits) for every word in the vocabulary.
#
# It finds the word with the highest score (.argmax(1)) to get the predicted word.
#
# Crucially, this predicted word becomes the inputs for the next loop iteration. This is the "no teacher forcing" part.
#
# This loop continues until the model predicts the <EOS> (End of Sequence) token or it reaches the max_len. Finally, it returns the sequence of generated word indices.

