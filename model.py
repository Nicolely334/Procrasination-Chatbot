import json
import pandas as pd
import string
import tensorflow
import numpy as np
import pandas as pd
import nltk
import random
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import *
from keras.metrics import sparse_categorical_crossentropy

with open("intents.json") as content:
    json_data = json.load(content)

tags = []
inputs = []
responses = {}

for intent in json_data['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

#converting to dataframe
data = pd.DataFrame({"patterns": inputs, "tags": tags})

#removing punctuations bcus model doesnt understand punctuation
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

#tokenize the data bcus model doesn't understand words it understands numbers-> Tensorflow's tokenizer assigns a unique token to each distinct word 
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

#apply padding to get all of the data to the same length as to send it to an rnn layer
x_train = pad_sequences(train)
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print(input_shape)

#define vocabulary
vocabulary = len(tokenizer.word_index)
print('number of unique words: ', vocabulary)
output_length = le.classes_.shape[0]
print('output length: ', output_length)

#creating the model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(i, x)

#compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training the model
train = model.fit(x_train, y_train, epochs=360)

while True:
    texts_p = []
    user_input = input('You: ')

    #removing punctuation and converting to lowercase
    user_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
    user_input = ''.join(user_input)
    texts_p.append(user_input)

    #tokenizing and padding
    user_input = tokenizer.texts_to_sequences(texts_p)
    user_input = np.array(user_input).reshape(-1)
    user_input = pad_sequences([user_input], input_shape)

    #getting output from model
    output = model.predict(user_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    print("Pluto: ", random.choice(responses[response_tag]))
    if response_tag == "goodbye":
        break
