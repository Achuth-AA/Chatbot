import random
import json
import nltk
from nltk import data
from model import chatbot_model
from train import bag_of_bords,tokenization
import torch
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data.json','r') as f:
    intents = json.load(f)

FILE = "trained_data.pth"
data = torch.load(FILE)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
words = data['words']
tags = data['tags']
model_state = data['model_state']

model = chatbot_model(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "AA Bot"
print("AA Bot is here : ")
while True:
    sentence = input('User: ')
    if sentence == "quit":
        break
    sentence = tokenization(sentence)
    x = bag_of_bords(sentence,words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _,predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output , dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry, i can't undrstand") 