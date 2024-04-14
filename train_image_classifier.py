import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsummary import summary

EPOCHS = 5000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.dropout_rate = 0.3
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(self.dropout_rate)

        self.pool0 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(self.dropout_rate)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(self.dropout_rate)


        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(self.dropout_rate)

        self.flatten = torch.nn.Flatten()
        self.relu5 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(256 * 40 * 40, 20)
        self.dropout5 = torch.nn.Dropout(self.dropout_rate)
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.pool0(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


augmentation = v2.Compose([
    v2.RandomResizedCrop(80, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    v2.RandomVerticalFlip(p=0.3),
    v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3),
    v2.ToTensor()
])

train_dataset = datasets.ImageFolder(root="data/train", transform=augmentation)
val_dataset = datasets.ImageFolder(root="data/val", transform=v2.Compose([v2.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ConvNet()
summary(model, (3, 80, 80), device='cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    if epoch == 100:
        pass
    # train model
    accuracy = 0
    train_loss = 0
    model.train()
    for i, train_batch in enumerate(train_loader):
        x, y = train_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_labels = (torch.sigmoid(y_hat) > 0.15).int()
        accuracy += torch.sum(pred_labels == y).item()

    accuracy /= len(train_dataset)

    if (epoch % 10 == 0):
        print(f"Epoch {epoch}, train_loss: {train_loss}, train accuracy: {accuracy}")

    # validate model
    missed = 0
    accuracy = 0
    model.eval()
    val_loss = 0
    for i, val_batch in enumerate(val_loader):
        x, y = val_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
        val_loss += loss.item()
        pred_labels = (torch.sigmoid(y_hat) > 0.15).int()
        accuracy += torch.sum(pred_labels == y).item()
        for i in range(len(y)):
            if pred_labels[i] != y[i] and y[i] == 1:
                missed += 1

    accuracy /= len(val_dataset)
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}, val_loss: {val_loss}, val accuracy: {accuracy}, missed: {missed} \n")


