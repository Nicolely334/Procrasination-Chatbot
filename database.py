import json
import pandas as pd

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

data = pd.DataFrame({"patterns": inputs, "tags": tags})
