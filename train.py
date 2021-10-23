import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open('intends.json', 'r') as f:
	intends = json.load(f)

all_words = []
tags = []
xy = []

for intend in intends['intents']:
	tag = intend['tag']
	tags.append(tag)
	for pattern in intend['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
# print(xy)
tags = sorted(set(tags))
# print(all_words)
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	X_train.append(bag)

	label = tags.index(tag)
	y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data= X_train
		self.y_data = y_train

	def __getitem__(self, idx):
		return self.x_data[idx], self.y_data[idx]

	def __len__(self):
		return self.n_samples


# Hyper parameters
batch_size = 8
input_size = len(all_words) # or x_train[0]
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1300
# print(input_size, len(all_words))
# print(output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = NeuralNet(input_size, hidden_size, output_size)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
	for (words, labels) in train_loader:
		words = words.to(device)
		labels = labels.to(dtype=torch.long).to(device)

		# forward
		outputs = model(words)
		loss = criterion(outputs, labels)

		# backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch+1) % 100 == 0:
		print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}')
print(f'final loss, loss={loss.item():.4f}')

data = {
	"model_state":model.state_dict(),
	"input_size": input_size,
	"output_size": output_size,
	"hidden_size": hidden_size,
	"all_words": all_words,
	"tags": tags
}


file_name = "data.pth"
torch.save(data, file_name)

print(f'Training Complete. File saved to {file_name}')



