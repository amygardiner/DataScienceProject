#library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torchcontrib.optim import SWA

filepath="/Users/amygardiner/Documents/University/PGD/Proj/Data/Paid_labelled/alldata.txt"
data = pd.read_csv(filepath, sep="\t")



#keeping only relevant columns and calculating sentence lengths
data = data[['tweet_text', 'label']]
data.columns=['text','label']
data['tweetlength'] = data['text'].apply(lambda x: len(x.split()))

#changing categories to numerical 
numericalmap = {'injured_or_dead_people':1/9, 'missing_trapped_or_found_people':2/9, 'displaced_people_and_evacuations':3/9, 'infrastructure_and_utilities_damage':4/9, 'donation_needs_or_offers_or_volunteering_services':5/9, 'caution_and_advice':6/9, 'sympathy_and_emotional_support':7/9, 'other_useful_information':8/9, 'not_related_or_irrelevant':1}
data=data.applymap(lambda s: numericalmap.get(s) if s in numericalmap else s)

tok = spacy.load('en_core_web_sm')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

#count number of occurences of each word
counts = Counter()
for index, row in data.iterrows():
    counts.update(tokenize(row['text']))
    
#deleting infrequent words
#print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
#print("num_words after:",len(counts.keys()))

#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)
    
def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length
data['encoded'] = data['text'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
#print(reviews.head())

X = list(data['encoded'])
y = list(data['label'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]
    
train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)

metricsfile = open('swaLSTMmetrics.txt', 'w')

def train_model(model, epochs=10, lr=0.001):
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        #optimizer.swap_swa_sgd()
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            s=("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f \n" % (sum_loss/total, val_loss, val_acc, val_rmse))
            metricsfile.write(s)

def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)

class LSTM_variable_input(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        
    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

model = LSTM_variable_input(vocab_size, 50, 50)
train_model(model, epochs=30, lr=0.1)  
metricsfile.close()  