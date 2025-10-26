#IMAGE CAPTIONING PROBLEM using RNNs for the Flickr 8k dataset -WITH TEACHER FORCING TRICK
#Some changes are made to the previous code to work with teacher forcing

#Loading the embeddings of the image and the captions
import torch.nn as nn
import torch
from collections import Counter
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#load the image embeddings first
# images=torch.load("image_embedding.pt") #images will be of dim 512, as the fc layer's dim was 512
# print(f"Loaded Image Embeddings from {len(images)} images.")
image_dict=torch.load("image_embedding.pt")
print(f"Loaded Image Embeddings from {len(image_dict)} images.")
# print(type(image_dict),)  # should be <class 'dict'>

# Convert dict → list of tensors
images = list(image_dict.values())
#
# print(type(images))  # <class 'list'>
# print(images[0].shape) #torch.Size([512])
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

class CaptionRNN(nn.Module):
    def __init__(self,vocab_size,hidden_dim,embed_dim):
        super(CaptionRNN,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad=True #as we are updating the weights of the embedding matrix

        self.rnn=nn.RNN(embed_dim+hidden_dim,hidden_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,vocab_size)
        self.img_fc=nn.Linear(512,hidden_dim)
        #image embedding dimensions should match the dimensions of the input going into the RNN as h0

    def forward(self,img_features,captions=None,max_len=20,teacher_forcing=True): #NOTE: This is changed
        batch_size=img_features.size(0)
        hidden=self.img_fc(img_features).unsqueeze(0) #shape will be (number_of_layers=1,batch_size,hidden_dim)
        #above is the h0
        img_embed_step=self.img_fc(img_features) #this will show the model image embedding at each step
        #feed the start token-<sos>
        inputs=torch.full((batch_size,1),word_to_index["<SOS>"],dtype=torch.long)
        logits_seq=[]

        for t in range(max_len):
            embedded=self.embedding(inputs).squeeze(1)
            rnn_input=torch.cat([embedded,img_embed_step],dim=1) #(batch_size,embed_dim+hidden_dim)
            rnn_input=rnn_input.unsqueeze(1)
            out,hidden=self.rnn(rnn_input,hidden)
            logits=self.fc(out.squeeze(1))
            logits_seq.append(logits.unsqueeze(1))
            predicted=logits.argmax(1).unsqueeze(1)

            #teacher forcing trick
            if teacher_forcing and captions is not None and t < captions.size(1):
                inputs=captions[:,t].unsqueeze(1) #use the ground truth value
            else:
                inputs=predicted #else use the predicted word from the previous timestep as input to the next timestep
        outputs=torch.cat(logits_seq,1)
        return outputs
hidden_dim = 256
model = CaptionRNN(embed_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=vocab_size)
print("CaptionRNN model created.")

criterion=nn.CrossEntropyLoss(ignore_index=word_to_index["<PAD>"])
epochs=50
optimizer=optim.Adam(model.parameters(),lr=0.001)
#train loop

for epoch in range(epochs):
    total_loss=0
    for batch_images,batch_captions ,_ in dataloader:
        optimizer.zero_grad()
        logits=model(batch_images,batch_captions,max_len=20,teacher_forcing=True)
        loss=criterion(logits.view(-1,vocab_size),batch_captions.view(-1))
        loss.backward() #calculate the gradients
        optimizer.step() #updates the weights
        total_loss+=loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")

#testing on one loop

#TESTING ON SINGLE IMAGE
# with torch.no_grad():
#     index=random.randint(0,len(dataset)-1)
#     sample_image,_,sample_name=dataset[index]
#     sample_image=sample_image.unsqueeze(0) #to add the dimension of the batch
#     logits=model(sample_image,max_len=20,teacher_forcing=False)
#     prediction=logits.argmax(2).tolist()[0]
#
#     caption=[]
#     for index in prediction:
#         word=index_to_word[index]
#         if word=="<EOS>":
#             break
#         if word != "<SOS>":
#             caption.append(word)
#     print(f"Sample (Image) name:{sample_name}")
#     print(f"Caption: {' '.join(caption)}")

# --- TESTING ON MULTIPLE IMAGES ---
model.eval()
import random

num_samples = 10  # number of images you want to test
sample_indices = random.sample(range(len(dataset)), num_samples)

print("\n--- Testing on 10 Random Images ---\n")
with torch.no_grad():
    for idx in sample_indices:
        sample_image, _, sample_name = dataset[idx]
        sample_image=sample_image.unsqueeze(0)  # add batch dimension
        logits=model(sample_image, max_len=20, teacher_forcing=False)
        prediction=logits.argmax(2).tolist()[0]

        caption=[]
        for index in prediction:
            word = index_to_word[index]
            if word == "<EOS>":
                break
            if word != "<SOS>":
                caption.append(word)

        print(f"Image: {sample_name}")
        print(f"Generated Caption: {' '.join(caption)}\n")










