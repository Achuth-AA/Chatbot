import json
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.utils.data.dataloader import DataLoader 
from model import chatbot_model



wordnet_lemma = WordNetLemmatizer()

def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def lemmatization(word):
    return wordnet_lemma.lemmatize(word.lower())

def bag_of_bords(sentence,words):
    sentence = [lemmatization(w) for w in sentence ]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx,w in enumerate(words):
        if w in sentence:
            bag[idx] = 1.0
    return bag


#loading th json data here
with open('data.json','r') as f :
    intents = json.load(f)

#debugging
#print(intents)

words = []
tags = []
label_list = []

for i in intents['intents']:
    tag = i['tag']
    tags.append(tag)
    for j in i['patterns']:
        word = tokenization(j)
        words.extend(word)
        label_list.append((word,tag))

stop_words = ['.',',','?','!',';']

words = [lemmatization(word) for word in words if word not in stop_words]
words = sorted(set(words))
tags = sorted(set(tags))
#print(words)

x_train = [] 
y_train = []

for (sentence,tag) in label_list:
    bag = bag_of_bords(sentence,words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

class Chatadata(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self,idx):
        return self.x_data[idx] , self.y_data[idx]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
input_size = len(x_train[0])
output_size = len(tags)
learning_rate = 0.001
tot_epochs= 2000

# print(input_size,len(words))
# print(output_size)
dataset = Chatadata()
train_loader = DataLoader(dataset = dataset , batch_size= batch_size , shuffle=True )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = chatbot_model(input_size , hidden_size , output_size ).to(device)

critieria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(tot_epochs):
    for(awords,alabels) in train_loader:
        awords = awords.to(device)
        alabels = alabels.to(device)
        outputs = model(awords)
        loss = critieria(outputs,alabels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{tot_epochs},loss={loss.item():.4f}')


trained_data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "words":words,
    "tags":tags
}

FILE = "trained_data.pth"
torch.save( trained_data , FILE)
print(f'model is saved at {FILE}')



