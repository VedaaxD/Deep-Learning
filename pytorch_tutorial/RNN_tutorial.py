import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn



df=pd.read_csv('100_Unique_QA_Dataset.csv')
# print(df.head())

#Step 1: tokenize
def tokenize(text):
    #preprocess the text
    text=text.replace('?','')
    text=text.replace('"','')
    return text.split()
# print(tokenize("What is the capital of France"))
#_________________________________________________________________________________________________________
#Step 2: Vocab
vocab={'<UNK>':0}

def build_vocab(row):
    # print(row['question'],row['answer'])
# df.apply(build_vocab,axis=1)
    tokenized_qn=tokenize(row['question'])
    tokenized_ans=tokenize(row['answer'])
    # print(tokenized_qn,tokenized_ans)
    merged_tokens=tokenized_qn+tokenized_ans
    print(merged_tokens)
    for token in merged_tokens:
        if token not in vocab:
            vocab[token]=len(vocab) #the new word will be added in the vocab if not present already

#build_vocab(df)# this is wrong
print(df.apply(build_vocab,axis=1))
#ro visualise how many words in the vocab
# print(len(vocab))
# print(vocab)
#____________________________________________________________________________________________________________
#Step 3: Word to Index
def text_to_indices(text,vocab):
    indexed_text=[]
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text

print(text_to_indices("What is Deep Learning?",vocab)) #here deep is not present in vocab as well as learning is not present
#in vocab so the index will be 0s
#Similary we are going to pass all the rows (qns+ans) from the dataset one by one to retrieve the indices

class QADataset(Dataset):
    #1st method
    def __init__(self,df,vocab):
        self.df=df
        self.vocab=vocab

    #2nd method
    def __len__(self):
        return self.df.shape[0]

    #3rd method
    def __getitem__(self, index): # getitem is used when -how to produce a single sample for a given index
        #text → integers → tensor is part of preparing that sample,
        #doing this inside the method ensures each sample is ready when requested by the DataLoader
        numerical_qn=text_to_indices(self.df.iloc[index]['question'],self.vocab)
        numerical_ans=text_to_indices(self.df.iloc[index]['answer'],self.vocab)

        return torch.tensor(numerical_qn),torch.tensor(numerical_ans[0]) #COnVERT TO TENSOR

#obj of the class
dataset=QADataset(df,vocab)

print(dataset.__getitem__(0)) #(tensor([1, 2, 3, 4, 5, 6]), tensor([7])) question1 and answer1
print(dataset[2]) #(tensor([10, 11, 12, 13, 14, 15]), tensor([16])) #qn 2 and ans 2

#obj of dataloader
dataloader=DataLoader(dataset,batch_size=1,shuffle=True) #since the data has few entries so padding may not be required

# for question,answer in dataloader:
#     print(question,answer)

#RNN Architecture
class SimpleRNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__() #call the parent constructor
        #make an embedding layer
        self.embedding=nn.Embedding(vocab_size,embedding_dim=50)
        self.rnn=nn.RNN(50,64,batch_first=True)
        self.linear=nn.Linear(64,vocab_size)
        pass


# #manually creating this tensor and embedding then fwd pass
# print(dataset[0][0]) #first row[0], first qn[0] if [1] then answer
# x=nn.Embedding(324,embedding_dim=50)
# print(x(dataset[0][0]).shape) #o/p torch.Size([6,50]) 6 vectors (words) , one vector of size 50
# a=x(dataset[0][0])
# y=nn.RNN(50,64)
#
# h_t,h_n=y(a) #all h_ts h_t=[h1,h2,h3,h4,h5,h6]
#
# print(h_t.shape) # all the timesteps output h1,h2,h3,h4,h5,h6
# print(h_t[0].shape) #o/p torch.Size([6,64])
# print(h_t[1].shape) #o/p torch.Size([1,64]) #only the last h_n here- h_6
#
# z=nn.Linear(64,324)
# final_layer_output=z(h_n)
# print(final_layer_output.shape) #o/p torch.Size([1, 324])


    def forward(self,question,):
        embedded_qn=self.embedding(question)
        output_seq,h_n=self.rnn(embedded_qn)
        final_output=self.linear(h_n[-1])
        return final_output


epochs=20

model=SimpleRNN(len(vocab)) #see init
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

#training loop
for epoch in range(epochs):
    total_loss=0

    for qn,ans in dataloader:
        optimizer.zero_grad()
        #forward pass
        output=model(qn)
        #loss
        loss=criterion(output,ans)
        #gradients
        loss.backward()
        #update
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch{epoch+1},Loss={total_loss:.4f}")






















