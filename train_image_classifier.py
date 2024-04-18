import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsummary import summary

torch.manual_seed(13)

EPOCHS = 250

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.dropout_rate = 0.6
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(self.dropout_rate)

        self.pool0 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same')
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(self.dropout_rate)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same')
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(self.dropout_rate)


        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same')
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(self.dropout_rate)

        self.flatten = torch.nn.Flatten()
        self.relu5 = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(256 * 40 * 40, 20)
        self.bn5 = torch.nn.BatchNorm1d(20)
        self.dropout5 = torch.nn.Dropout(self.dropout_rate)
        self.fc2 = torch.nn.Linear(20, 1)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn5(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


augmentation = v2.Compose([
    v2.RandomResizedCrop(80, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    v2.RandomVerticalFlip(p=0.3),
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomGrayscale(p=0.1),
    #v2.RandomPerspective(distortion_scale=0.2, p=0.2),
    #v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    v2.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    v2.GaussianBlur(kernel_size=3),
    v2.ToTensor()
])

if __name__ == "__main__":

    train_dataset = datasets.ImageFolder(root="data/train", transform=augmentation)
    val_dataset = datasets.ImageFolder(root="data/val", transform=v2.Compose([v2.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ConvNet()
    summary(model, (3, 80, 80), device='cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = np.inf
    best_train_loss = np.inf
    best_model = None
    best_epoch = 0
    best_accuracy = 0
    best_train_accuracy = 0

    for epoch in range(EPOCHS):
        if epoch == 100:
            pass
        # train model
        train_accuracy = 0
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
            pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
            train_accuracy += torch.sum(pred_labels == y).item()

        train_accuracy /= len(train_dataset)

        if (epoch % 1 == 0):
            print(f"Epoch {epoch}, train_loss: {train_loss}, train accuracy: {train_accuracy}")

        # validate model
        missed = 0
        false_positives = 0
        eval_accuracy = 0
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
            pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
            eval_accuracy += torch.sum(pred_labels == y).item()
            for i in range(len(y)):
                if pred_labels[i] != y[i] and y[i] == 1:
                    missed += 1
                if pred_labels[i] != y[i] and y[i] == 0:
                    false_positives += 1

        eval_accuracy /= len(val_dataset)

        #if (eval_accuracy > best_accuracy) or (eval_accuracy == best_accuracy and val_loss < best_val_loss and train_accuracy > best_train_accuracy):
        if val_loss < best_val_loss and train_loss < best_train_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            best_epoch = epoch
            best_accuracy = eval_accuracy
            best_train_loss = train_loss
            best_train_accuracy = train_accuracy
            best_missed = missed

        if (epoch % 1 == 0):
            print(f"Epoch {epoch}, val_loss: {val_loss}, val accuracy: {eval_accuracy}, missed: {missed}, false positives: {false_positives} \n")


    torch.save(best_model, "image_classifier.pkl")
    print(f"Best model at epoch {best_epoch}, val_loss: {best_val_loss}, val accuracy: {best_accuracy} with train loss: {best_train_loss} and train accuracy: {best_train_accuracy} \
        with missed: {best_missed} and false positives: {false_positives} \n")


    model2 = ConvNet()
    model2.load_state_dict(torch.load("image_classifier.pkl"))
    model2.eval()
    model2.to(device)
    missed = 0
    false_positives = 0
    for val_batch in train_loader:
        x, y = val_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model2(x)
        pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
        for i in range(len(y)):
            if pred_labels[i] != y[i] and y[i] == 1:
                missed += 1
            if pred_labels[i] != y[i] and y[i] == 0:
                false_positives += 1

    print(f"Final missed: {missed}, false positives: {false_positives}")

    missed = 0
    false_positives = 0
    for val_batch in val_loader:
        x, y = val_batch
        y = y.unsqueeze(1).float()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model2(x)
        pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
        for i in range(len(y)):
            if pred_labels[i] != y[i] and y[i] == 1:
                missed += 1
            if pred_labels[i] != y[i] and y[i] == 0:
                false_positives += 1

    print(f"Final missed: {missed}, false positives: {false_positives}")