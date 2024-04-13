import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsummary import summary

EPOCHS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.7)

        self.pool0 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same')
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.7)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(0.7)

        self.flatten = torch.nn.Flatten()
        self.relu5 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(32 * 20 * 20,16)
        self.relu6 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(0.7)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.pool0(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


augmentation = v2.Compose([
    v2.RandomResizedCrop(80, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    v2.RandomVerticalFlip(p=0.7),
    v2.RandomRotation(30),
    v2.ColorJitter(brightness=0.2, hue=0.1, contrast=0.1, saturation=0.1),
    v2.GaussianBlur(3, sigma=(0.1, 2.0)),
    v2.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    lambda x: x/255.0
])

train_dataset = datasets.ImageFolder(root="data/train", transform=augmentation)
val_dataset = datasets.ImageFolder(root="data/val", transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ConvNet()
#summary(model, (3, 80, 80), device='cpu')
#exit()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    # train model
    accuracy = 0
    model.train()
    for i, train_batch in enumerate(train_loader):
        x, y = train_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_labels = torch.sigmoid(y_hat).round().int()
        accuracy += torch.sum(pred_labels == y).item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch}, batch {i}, loss: {loss.item()}")
    accuracy /= len(train_dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch}, train accuracy: {accuracy}")

    # validate model
    missed = 0
    model.eval()
    for i, val_batch in enumerate(val_loader):
        x, y = val_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        pred_labels = torch.sigmoid(y_hat).round().int()
        accuracy += torch.sum(pred_labels == y).item()
        for i in range(len(y)):
            if pred_labels[i] != y[i] and y[i] == 1:
                missed += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch}, batch {i}, validation loss: {loss.item()}")

    accuracy /= len(val_dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch}, val accuracy: {accuracy}, missed: {missed}")


