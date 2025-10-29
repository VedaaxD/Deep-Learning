#Implement sequence to sequence learning using LSTM and GRU. You can use the the English to Hindi & Hindi to English
# corpus dataset - https://www.kaggle.com/code/aiswaryaramachandran/english-to-hindi-neural-machine-translation

#Implementing the sequence to sequence learning using LSTMs
import torch
import pandas as pd
import re
from collections import Counter
from torch import nn
from sklearn.model_selection import train_test_split
from statsmodels.datasets.utils import Dataset
from torch.utils.data import DataLoader



pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',500)
lines=pd.read_csv('Hindi_English_Truncated_Corpus.csv')


#DATA PREPROCESSING
print("The no of sources present in this corpus are:")
print(lines['source'].value_counts()) #how many different sources are present

# lines=lines[lines['source']=='ted']
# print()(lines.head(20))
print('--------------------------------------------------')
print(f"Checking how many null values are present in the corpus:")
print(pd.isnull(lines).sum()) #2 rows in the english_sentence col is null

print('--------------------------------------------------')
print(lines[pd.isnull(lines['english_sentence'])]) #identify the rows which have NaN value

#dropping the duplicates from the csv from the filtered samples (which doesn't have NaN)
lines=lines[~pd.isnull(lines['english_sentence'])]
# lines=lines.drop_duplicates(inplace=True) #modifying it inplace and assigning into lines gave error
lines.drop_duplicates(inplace=True)

#make all the english_sentence to lowercase
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())

#remove punctuations
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(r"[^\w\s]","",x))
lines['hindi_sentence']=lines["hindi_sentence"].apply(lambda x: re.sub(r"[^\w\s]","",x))
#r"[^\w\s]" - regex for removing all the char except the words and spaces

#removing numbers
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(r"\d+","",x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(r"\d+","",x))

#add the start and end tokens to all the target sentences
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: "START_" + x + "_END")
print(lines.head(10))


#BUILDING VOCABULARY
def build_vocab(sentences,min_frq=2):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    counter=Counter()
    for s in sentences:
        counter.update(s.split())
    for word,frq in counter.items():
        if frq >= min_frq:
            vocab[word] =len(vocab)
    idx2word={idx:word for word,idx in vocab.items()}
    return vocab,idx2word

def encode(sentence,vocab):
    return [vocab.get(w,vocab["<UNK>"]) for w in sentence.split()] #for every word in the sentence it gets the index, if present
#in the vocab, otherwise returns the index for the unknown

def pad_sequences(sequences, pad_index):
    max_len = max(len(s) for s in sequences)
    padded = [torch.cat([s, torch.full((max_len - len(s),), pad_index, dtype=torch.long)]) for s in sequences]
    return torch.stack(padded)

eng_vocab,eng_idx2word=build_vocab(lines['english_sentence'])
hin_vocab,hin_idx2word=build_vocab(lines['hindi_sentence'])
print(f"English vocab size: {len(eng_vocab)}, Hindi vocab size: {len(hin_vocab)}")


#TRAIN TEST SPLIT
pairs=list(zip(lines["english_sentence"],lines["hindi_sentence"]))
train_pairs,test_pairs=train_test_split(pairs,test_size=0.1,random_state=42)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

# --- Translation function ---
def translate_sentence(model, sentence, source_vocab, target_vocab, target_idx2word, max_len=20):
    model.eval()
    src = torch.tensor(encode(sentence.lower(), source_vocab), dtype=torch.long).unsqueeze(0)
    hidden = model.encoder(src)
    input_token = torch.tensor([target_vocab["START_"]], dtype=torch.long)  # start token
    outputs = []

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            top1 = output.argmax(1)
            word = target_idx2word[top1.item()]
            if word == "_END":
                break
            outputs.append(word)
            input_token = top1

    return " ".join(outputs)


# --- BLEU score calculation ---
def evaluate_bleu(model, test_pairs, source_vocab, target_vocab, target_idx2word, n_samples=100):
    smoothie = SmoothingFunction().method4
    scores = []

    for i, (eng, hin) in enumerate(test_pairs[:n_samples]):  # evaluate on subset for speed
        pred = translate_sentence(model, eng, source_vocab, target_vocab, target_idx2word)
        ref = hin.replace("START_", "").replace("_END", "").split()
        pred_tokens = pred.split()
        bleu = sentence_bleu([ref], pred_tokens, smoothing_function=smoothie)
        scores.append(bleu)

    avg_bleu = sum(scores) / len(scores)
    print(f"Average BLEU score on {n_samples} samples: {avg_bleu:.4f}")
    return avg_bleu

#DATASET AND DATALOADER
class TranslationDataset(Dataset):
    def __init__(self,pairs,source_vocab,target_vocab):
        self.pairs=pairs
        self.source_vocab=source_vocab
        self.target_vocab=target_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        src_ids = torch.tensor(encode(source, self.source_vocab), dtype=torch.long)
        trg_ids = torch.tensor(encode(target, self.target_vocab), dtype=torch.long)
        return src_ids, trg_ids

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequences(src_batch, pad_index=0)
    trg_padded = pad_sequences(trg_batch, pad_index=0)
    return src_padded, trg_padded

train_dataset=TranslationDataset(train_pairs,eng_vocab,hin_vocab)
test_dataset=TranslationDataset(test_pairs,eng_vocab,hin_vocab)
train_loader=DataLoader(train_dataset,batch_size=64,collate_fn=collate_fn)
test_loader=DataLoader(test_dataset,batch_size=64,collate_fn=collate_fn)



#SEQ2SEQ model
class Encoder(nn.Module):
    def __init__(self,input_dim,embed_dim,hidden_dim,model_type="lstm"):
        super().__init__()
        self.embedding=nn.Embedding(input_dim,embed_dim,padding_idx=0)
        self.model_type=model_type
        if model_type == "lstm":
            self.rnn=nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        else:
            self.rnn=nn.GRU(embed_dim,hidden_dim,batch_first=True)
    def forward(self,x):
        emb=self.embedding(x)
        outputs,hidden=self.rnn(emb)
        return hidden

class Decoder(nn.Module):
    def __init__(self,output_dim,embed_dim,hidden_dim,model_type="lstm"):
        super().__init__()
        self.embedding=nn.Embedding(output_dim,embed_dim,padding_idx=0)
        self.model_type=model_type
        if model_type == "lstm":
            self.rnn=nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        else:
            self.rnn=nn.GRU(embed_dim,hidden_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,output_dim)

    def forward(self,x,hidden):
        x=x.unsqueeze(1)
        emb=self.embedding(x)
        if self.model_type == "lstm":
            output,(h,c)=self.rnn(emb,hidden)
            pred=self.fc(output.squeeze(1))
            return pred,(h,c)
        else:
            output,h=self.rnn(emb,hidden)
            pred=self.fc(output.squeeze(1))
            return pred,h

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        batch_size,trg_len=trg.shape
        vocab_size=self.decoder.fc.out_features
        outputs=torch.zeros(batch_size,trg_len,vocab_size)
        hidden=self.encoder(src)
        input_token=trg[:,0]

        for t in range(1,trg_len):
            output,hidden=self.decoder(input_token,hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs


# --- TRAIN AND EVALUATE FUNCTION ---
def train_and_evaluate(model, train_loader, test_pairs, eng_vocab, hin_vocab, hin_idx2word,
                       criterion, optimizer, n_epochs=5, teacher_forcing_ratio=0.5, max_len=20):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate BLEU after training
    avg_bleu = evaluate_bleu(model, test_pairs, eng_vocab, hin_vocab, hin_idx2word, n_samples=100)
    return avg_bleu


# --- TRAIN LSTM ---
encoder_lstm = Encoder(len(eng_vocab), 256, 512, model_type="lstm")
decoder_lstm = Decoder(len(hin_vocab), 256, 512, model_type="lstm")
model_lstm = Seq2Seq(encoder_lstm, decoder_lstm)
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("Training LSTM...")
bleu_lstm = train_and_evaluate(model_lstm, train_loader, test_pairs, eng_vocab, hin_vocab, hin_idx2word,
                               criterion, optimizer_lstm, n_epochs=5)

# --- TRAIN GRU ---
encoder_gru = Encoder(len(eng_vocab), 256, 512, model_type="gru")
decoder_gru = Decoder(len(hin_vocab), 256, 512, model_type="gru")
model_gru = Seq2Seq(encoder_gru, decoder_gru)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=1e-3)

print("\nTraining GRU...")
bleu_gru = train_and_evaluate(model_gru, train_loader, test_pairs, eng_vocab, hin_vocab, hin_idx2word,
                              criterion, optimizer_gru, n_epochs=5)

# --- COMPARE BLEU SCORES ---
print("\nAverage BLEU Scores:")
print(f"LSTM: {bleu_lstm:.4f}")
print(f"GRU: {bleu_gru:.4f}")

if bleu_lstm > bleu_gru:
    print("LSTM performs better")
elif bleu_gru > bleu_lstm:
    print("GRU performs better")
else:
    print("Both models perform equally")

#This code takes a very long time to run, I have optimized this code in separate sections LSTM and GRU
#Additionally performed nmt using transformers
