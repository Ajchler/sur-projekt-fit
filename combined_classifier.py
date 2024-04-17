from audio_gmm import train_gmm, load_gmm, read_dataset, AudioDataset, Pipeline
from train_image_classifier import ConvNet
import numpy as np
from torchvision import datasets, transforms
import torch

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

        self.fc1 = torch.nn.Linear(2, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 1)

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

EPOCHS = 100

model = CombinedClassifier()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
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

val_audio_non_target = AudioDataset("data/val/non_target_val", transform=pipeline)
val_audio_target = AudioDataset("data/val/target_val", transform=pipeline)
audio_data_non_target_val = []
audio_data_target_val = []
audio_data_non_target_train = [x[1] for x in read_dataset(val_audio_non_target, train=False)]
audio_data_target_val = [x[1] for x in read_dataset(val_audio_target, train=False)]


train_image_dataset = datasets.ImageFolder(root="data/train", transform=transforms.Compose([transforms.ToTensor()]))
train_image_dataset = torch.utils.data.ConcatDataset([train_image_dataset] * 1)
train_images = [x[0].numpy() for x in train_image_dataset]
targets_train = [x[1] for x in train_image_dataset]

val_image_dataset = datasets.ImageFolder(root="data/val", transform=transforms.Compose([transforms.ToTensor()]))
train_image_dataset = torch.utils.data.ConcatDataset([val_image_dataset] * 1)
val_images = [x[0].numpy() for x in val_image_dataset]
targets_val = [x[1] for x in val_image_dataset]

cur_pos = 0
for i in range(EPOCHS):
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
        audio_batch = audio_batch
        image_batch = torch.tensor(image_batch)
        y_hat = model(audio_batch, image_batch)
        y = torch.tensor(shuffled_targets[cur_pos:cur_pos + 32], dtype=torch.float32).view(-1, 1)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_pos += 32

    model.eval()


