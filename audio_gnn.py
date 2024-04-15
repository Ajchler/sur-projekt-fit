import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from glob import glob

class AudioDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = glob(root + '/*.wav')
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torchaudio.load(self.files[idx])
        if self.transform:
            data = self.transform(data)
        return data

class TrimSilence(torch.nn.Module):
    # Trim silence parts from the beginning and the end of audio sample
    def __init__(self, start_size, start_thr, end_size, end_thr):
        super(TrimSilence, self).__init__()
        # https://stackoverflow.com/a/72426502
        self.start_size = start_size
        self.start_thr = start_thr
        self.end_size = end_size
        self.end_thr = end_thr
        
    def forward(self, x):
        # Using sox to trim the audio
        _sox_trim_silence = [
            ['silence', '1', str(self.start_size), str(self.start_thr) + '%'],
            ['reverse'],
            ['silence', "1", str(self.end_size), str(self.end_thr) + '%'],
            ['reverse']
        ]
        return torchaudio.sox_effects.apply_effects_tensor(x[0], x[1], _sox_trim_silence)

pipeline = torch.nn.Sequential(
    TrimSilence(0.2, 1, 0.5, 0.2),
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
)

train_target = AudioDataset('data/target_train', transform=pipeline)
train_target_loader = DataLoader(train_target, batch_size=1, shuffle=False)

for i, train_batch in enumerate(train_target_loader):
    torchaudio.save('test/' + str(i) + '.wav', train_batch[0][0], train_batch[1][0])
    exit()
    
# ["silence", "1", "0.2", "1%"],
# ["reverse"],
# ["silence", "1", "0.5", "0.2%"],
# ["reverse"]