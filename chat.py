import random
import json
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intends.json', 'r') as f:
	intents = json.load(f)


file_name = "data.pth"
data = torch.load(file_name)


input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"
print("Let's chat! type 'quit' to quit")


def get_response(sentence):

	sentence = tokenize(sentence)
	X = bag_of_words(sentence, all_words)
	X = X.reshape(1, X.shape[0])
	X = torch.from_numpy(X)

	output = model(X)
	_, predicted = torch.max(output, dim=1)

	tag = tags[predicted.item()]

	probs = torch.softmax(output, dim=1)
	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents["intents"]:
			if tag == intent["tag"]:
				return random.choice(intent['responses'])

	return "I do not understand...."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)






