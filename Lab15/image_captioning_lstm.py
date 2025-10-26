##Implement LSTM using PyTorch for image captioning

# IMAGE CAPTIONING PROBLEM using LSTMs for the Flickr 8k dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
image_dict=torch.load("image_dict.pt")
print(f"Loaded Image Embeddings from {len(image_dict)} images.")

#Convert dict to a list of tensors
images=list(image_dict.values())
image_names=list(image_dict.keys())

#load the captions
with open("captions.txt" , "r", encoding="utf-8") as f:
    captions=f.readlines()
print(f"Loaded {len(captions)} captions.")

#preprocess the captions
processed_captions=[]
for line in captions:
    if ',' in line: #this will not consider the image name (xxx17291.jpg) as a caption which will sit in our vocab
        _,caption_text=line.strip().split(',',1)
    else:
        caption_text=line.strip()
    caption_text=caption_text.lower()
    processed_captions.append(caption_text)

# build vocabulary
#Tokenize all the captions into words
tokens=[]
for line in processed_captions:
    for word in line.strip().split():
        word=word.lower()
        tokens.append(word) #also can be simplified as list comprehension - [word.lower() for line in captions for word in line.strip().split()]

#count word frequencies
word_counts=Counter(tokens)

#Special tokens
special_tokens=["<SOS>","<EOS>","<UNK>","<PAD>"]
vocab=special_tokens+list(word_counts.keys()) #all the words in the vocabulary
vocab_size=len(vocab)
print(f"Vocabulary Size: {vocab_size}")
# print(f"Vocab:{vocab}") #to check if the vocab doesn't have proper collection of words
word_to_index={word:idx for idx, word in enumerate(vocab)}
index_to_word={idx:word for word,idx in word_to_index.items()}

#using the GloVe embeddings with 50 dimensions
embed_dim=50
embedding_matrix=torch.randn(vocab_size,embed_dim)
#we override the embedding_matrix if the word in our captions is present in glove, if not use the random value initialized.
glove_path="/home/ibab/Downloads/glove.2024.wikigiga.50d/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values=line.strip().split()
        word=values[0]
        if len(values)!=embed_dim + 1:
            continue #skip malformed line
        try:
            vector = np.array(values[1:], dtype=np.float32)
        except ValueError:
            continue  #skip lines where conversion fails

        if word in word_to_index:
            embedding_matrix[word_to_index[word]]=torch.from_numpy(vector) #overrides the random initialization


def captions_to_indices(captions, word_to_index, max_len=20):
    indexed_captions=[]
    for cap in captions:
        words=cap.strip().split()
        indices=[word_to_index["<SOS>"]] +[word_to_index.get(w,word_to_index["<UNK>"])for w in words]+[word_to_index["<EOS>"]]
        #pad if the length is lesser
        if len(indices)<max_len:
            indices+=[word_to_index["<PAD>"]] * (max_len-len(indices))
        else:
            indices=indices[:max_len]
        indexed_captions.append(indices)
    return torch.tensor(indexed_captions, dtype=torch.long)
caption_indices=captions_to_indices(processed_captions, word_to_index)

#make a pytorch dataset class
class CaptionDataset(Dataset):
    def __init__(self, images, captions,image_names):
        self.images = images
        self.captions = captions
        self.image_names = image_names
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx].float(), self.captions[idx].long(),self.image_names[idx]

dataset = CaptionDataset(images, caption_indices,image_names)
batch_size=64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class CaptionLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim):
        super(CaptionLSTM,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad=True #as we are updating the weights of the embedding matrix
        self.lstm=nn.LSTM()
