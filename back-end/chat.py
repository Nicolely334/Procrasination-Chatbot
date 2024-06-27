import random
import json

import torch

from model import NeuralNet
from nltk_fns import bag_of_words, tokenize

with open('./back-end/intents.json') as fl:
    intents = json.load(fl)

data = torch.load("model.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

#new instance of nn with stats of the pre-trained model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval() #disables trianing mode

def get_response(msg):
    """
    takes the user message and returns the bot's response
    """
    sentence = tokenize(msg)
    #print(sentence)
    bog = bag_of_words(sentence, all_words)
    bog = bog.reshape(1, bog.shape[0])
    bog = torch.from_numpy(bog)

    output = model(bog)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print("prob: ", prob.item())
    if prob.item() > 0.15: #here work up
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry, I don't understand. Could you say that again?"


if __name__ == "__main__":
    print("My name is Pluto! Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        response = get_response(sentence)
        print("Pluto:", response)