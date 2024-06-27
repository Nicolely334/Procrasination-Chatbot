import json
import numpy
import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralNet
from nltk_fns import tokenize, bag_of_words, stem

ignore_words = ['?', '.', '!', ',']

with open("./back-end/intents.json") as f:
    intents = json.load(f)

all_words = []
tags = []
tagged_word_dictionary = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        tagged_word_dictionary.append((w, tag))

print('here')
#stem and lower each word
all_words = [stem(w) for w in all_words if w not in ignore_words]
#remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#training data
X_train = []
y_train = []
for (sentence, tag) in tagged_word_dictionary:
    #X: bag of words for each sentence
    bag = bag_of_words(sentence, all_words)
    X_train.append(bag)
    #y:classification labels
    label = tags.index(tag)
    y_train.append(label)

X_train = numpy.array(X_train)
y_train = numpy.array(y_train)

print('here')
#hyper parameters 
num_epochs = 87 #100 epochs-> 100, 60 -> > 75 already
batch_size = 8 # 7 close
learning_rate = 0.001 #0.1, 0.01, 0.015, 0.001, 0.003 
input_size = len(X_train[0])
hidden_size = 8 #7 close
output_size = len(tags)

train_loader = DataLoader(dataset=TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True, num_workers=0)

model = NeuralNet(input_size, hidden_size, output_size)

#accuracy/loss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #try

for epoch in range(num_epochs):
    correct = 0
    for (words, labels) in train_loader:
        #first, forward pass
        outputs = model(words)
        labels = labels.to(dtype=torch.long)
        _, predicted = torch.max(outputs, dim=1)
        
        #going back in and optimzing
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (predicted == labels).float().sum()
    accuracy = 100 * correct / len(X_train)
    print(f'epoch {epoch+1}, Accuracy: {accuracy:.3f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

torch.save(data, "model.pth")