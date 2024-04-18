from audio_gmm import train_gmm, load_gmm, read_dataset, AudioDataset, Pipeline
from train_image_classifier import ConvNet
import numpy as np
from torchvision import datasets, transforms
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

class AudioClassifier(torch.nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.t_gmm, self.nt_gmm = load_gmm()

    def forward(self, x):
        scores = []
        for sample in x:
            scores.append(sum(self.t_gmm.score_samples(sample)) - sum(self.nt_gmm.score_samples(sample)))

        return torch.tensor(scores, dtype=torch.float32)

class CombinedClassifier(torch.nn.Module):
    def __init__(self):
        super(CombinedClassifier, self).__init__()
        self.audio_classifier = AudioClassifier()
        self.image_classifier = ConvNet()
        self.image_classifier.load_state_dict(torch.load('image_classifier.pkl'))
        for param in self.image_classifier.parameters():
            param.requires_grad = False
        self.image_classifier.eval()

        self.fc1 = torch.nn.Linear(2, 100)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, audio_batch, image_batch):
        x1 = self.audio_classifier(audio_batch)
        x1 = x1.view(-1, 1)
        x1 = torch.tanh(x1)
        x2 = self.image_classifier(image_batch)
        x = torch.cat((x1, x2), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

EPOCHS = 60

model = CombinedClassifier()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
pipeline = Pipeline()
train_audio_non_target = AudioDataset("data/train/non_target_train", transform=pipeline)
train_audio_target = AudioDataset("data/train/target_train", transform=pipeline)
audio_data_non_target_train = []
audio_data_target_train = []
for i in range(1):
    a_ntarget = read_dataset(train_audio_non_target, train=False)
    audio_data_non_target_train += ([x[1] for x in a_ntarget])
    a_target = read_dataset(train_audio_target, train=False)
    audio_data_target_train += ([x[1] for x in a_target])
combined_audio_data_train = audio_data_non_target_train + audio_data_target_train

val_audio_non_target = AudioDataset("data/val/non_target_dev", transform=pipeline)
val_audio_target = AudioDataset("data/val/target_dev", transform=pipeline)
audio_data_non_target_val = []
audio_data_target_val = []
audio_data_non_target_val = [x[1] for x in read_dataset(val_audio_non_target, train=False)]
audio_data_target_val = [x[1] for x in read_dataset(val_audio_target, train=False)]
combined_audio_data_val = audio_data_non_target_val + audio_data_target_val


train_image_dataset = datasets.ImageFolder(root="data/train", transform=transforms.Compose([transforms.v2.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_image_dataset, batch_size=32, shuffle=True)
train_image_dataset = torch.utils.data.ConcatDataset([train_image_dataset] * 1)
train_images = [x[0].numpy() for x in train_image_dataset]
targets_train = [x[1] for x in train_image_dataset]

val_image_dataset = datasets.ImageFolder(root="data/val", transform=transforms.v2.Compose([transforms.v2.ToTensor()]))
val_image_dataset = torch.utils.data.ConcatDataset([val_image_dataset] * 1)
val_images = [x[0].numpy() for x in val_image_dataset]
targets_val = [x[1] for x in val_image_dataset]


for i in range(EPOCHS):
    cur_pos = 0
    train_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    val_loss = 0
    shuffled_audio = []
    shuffled_images = []
    shuffled_targets = []
    idx = np.random.permutation(len(combined_audio_data_train))
    for j in range(len(idx)):
        shuffled_audio.append(combined_audio_data_train[idx[j]])
        shuffled_images.append(train_images[idx[j]])
        shuffled_targets.append(targets_train[idx[j]])

    model.train()
    while cur_pos <= len(idx):
        audio_batch = combined_audio_data_train[cur_pos:cur_pos + 32]
        image_batch = train_images[cur_pos:cur_pos + 32]
        image_batch = torch.tensor(image_batch)
        y_hat = model(audio_batch, image_batch)
        pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
        y = torch.tensor(shuffled_targets[cur_pos:cur_pos + 32], dtype=torch.float32).view(-1, 1)
        loss = loss_fn(y_hat, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_pos += 32
        train_accuracy += torch.sum(pred_labels == y).item()

    train_accuracy /= len(combined_audio_data_train)

    print(f"Epoch {i}, train_loss: {train_loss}, train accuracy: {train_accuracy}")

    cur_pos = 0
    missed = 0
    model.eval()
    while cur_pos <= len(val_images):
        audio_batch = combined_audio_data_val[cur_pos:cur_pos + 32]
        image_batch = val_images[cur_pos:cur_pos + 32]
        image_batch = torch.tensor(image_batch)
        y = torch.tensor(targets_val[cur_pos:cur_pos + 32], dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            y_hat = model(audio_batch, image_batch)
            loss = loss_fn(y_hat, y)
        pred_labels = (torch.sigmoid(y_hat) > 0.2).int()
        val_loss += loss.item()
        cur_pos += 32
        val_accuracy += torch.sum(pred_labels == y).item()
        for j in range(len(y)):
            if pred_labels[j] != y[j] and y[j] == 1:
                missed += 1

    val_accuracy /= len(combined_audio_data_val)

    print(f"Epoch {i}, val_loss: {val_loss}, validation accuracy: {val_accuracy}, missed: {missed}\n")


