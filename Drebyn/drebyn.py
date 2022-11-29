from cgi import test
import string
from turtle import forward
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
import numpy as np
import ijson
import pandas as pd
import os, os.path
from sklearn import preprocessing
from sklearn.feature_extraction.text import HashingVectorizer

# Set seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class DrebynDataset(Dataset):
    def __init__(self):
        super(DrebynDataset).__init__()
        self.pathX = os.path.join(os.getcwd(), "datasets", "drebin-combined-X.json")
        self.pathY = os.path.join(os.getcwd(), "datasets", "drebin-combined-Y.json")
        
        self.X = torch.tensor(self.parseX(self.pathX))
        self.Y = torch.tensor(self.parseY(self.pathY))

        self.X = self.X.view(-1, 1)
        self.Y = self.Y.view(-1, 1)

    def __getitem__(self, index):
        pass
        return super().__getitem__(index)

    def __len__(self):
        pass
        return None

    def parseX(self, path):
        with open(path, "r") as file:
            features = ijson.items(file, 'item')
            
            # size = 259230
            dataset = [str] * 259230
            for index, feature in enumerate(features):
                dataset[index] = str(feature)
        
        # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
        # tokenise the strings
        
        vectoriser = HashingVectorizer(n_features=259230)
        target = vectoriser.fit_transform(dataset)
        #label_encoder = preprocessing.LabelEncoder()
        #target = label_encoder.fit_transform(dataset[:50])

        print(target.shape)


        return target
    
    def parseY(self, path):
        with open(path, "r") as file:
            content = file.read()

            # just making sure
            content = content.strip()
            
            # remove brackets, spaces, and convert to an array
            content = content[1: (len(content)-1)]
            content = content.replace(" ", "")
            content = content.split(",")
            
            labels = [int(y) for y in content]

            return labels[:50]

class SVM(nn.Module):
    # http://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/
    # https://www.youtube.com/watch?v=UX0f9BNBcsY

    def __init__(self, n_features):
        super(SVM, self).__init__()

        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

# Create Tensor dataset and Dataloader

dataset = DrebynDataset()

train_dataset = TensorDataset(dataset.X, dataset.Y)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_model = SVM(dataset.X.shape[1])

"""
# hyperparams and optimiser
epochs = 1000
C = 1.0
lr = 0.001

optimiser = torch.optim.Adam(test_model.parameters(), lr=lr)

total_loss = 0
n_batch = 0

test_model.train()

for epoch in range(epochs):
    for batch in train_dataloader:
        optimiser.zero_grad()

        train_x, train_y = batch
        output = test_model(train_x.float())
        loss = 0.5 * torch.norm(test_model.linear.weight.squeeze()) ** 2
        loss += C * torch.clamp(1 - train_y*output, min=0).mean()

        loss.backward()
        optimiser.step()

        total_loss += loss
        n_batch += 1
    
    print(f"Epoch: {epoch}\t Loss: {total_loss / n_batch}")

"""