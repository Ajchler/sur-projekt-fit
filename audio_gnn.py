import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

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
        audio = torchaudio.load(self.files[idx])
        if self.transform:
            audio = self.transform(audio)
        return {"name": self.files[idx], "data": audio}

class Pipeline(torch.nn.Module):
    def __init__(self, in_freq=16000, out_freq=16000, n_fft=256, win_length=200, win_hop=100, n_mels=23, n_mfcc=13, debug=False):
        # win_length = number of samples in each window - for 200 samples you get 12.5ms windows at 16kHz
        # win_hop = number of samples between the start of two consecutive windows
        # n_fft = number of samples in each window (at 256 with win_length 200 you get 56 zero values at the end of each window)
        # n_mels = number of mel filterbanks
        # n_mfcc = number of mfcc coefficients
        super(Pipeline, self).__init__()
        
        self.debug = debug
        self.out_freq = out_freq
        
        self.resample = torchaudio.transforms.Resample(orig_freq=in_freq, new_freq=out_freq)
        self.trim_params = [["silence", "1", "0.2", "1%"], ["reverse"], ["silence", "1", "0.5", "0.2%"], ["reverse"]]
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=win_hop)
        # Data augmentation goes here
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, n_stft=n_fft // 2 + 1, sample_rate=out_freq)
        self.dct = torchaudio.functional.create_dct(n_mfcc=n_mfcc, n_mels=n_mels, norm='ortho')
        # self.mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, sample_rate=out_freq, log_mels=True, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": win_hop, "win_length": win_length})
        
    def forward(self, x):
        waveform, sample_rate = x
        
        # Resample the audio
        resampled = self.resample(waveform)
        resampled_rate = self.out_freq
        # Trim silent parts from the beginning and the end of the audio
        trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(resampled, resampled_rate, self.trim_params)
        if self.debug:
            torchaudio.save('trimmed.wav', trimmed, resampled_rate)
        # Compute the spectrogram
        spec = self.spec(trimmed)
        # Spectogram augmentation (none for now)
        spec_augmented = spec
        # Convert to mel-scale
        mel = self.mel_scale(spec_augmented)
        # Get MFCC coefficients
        mfcc = torch.matmul(torch.log(mel[0].T), self.dct)
        
        # Plot graphs
        if self.debug:
            fig, axs = plt.subplots(3, 1)
            fig.set_figheight(15)
            axs[0].plot(np.linspace(0, len(waveform[0]) / sample_rate, num=len(waveform[0])), waveform[0].numpy())
            axs[1].plot(np.linspace(0, len(trimmed[0]) / resampled_rate, num=len(trimmed[0])), trimmed[0].numpy())
            axs[2].plot(np.linspace(0, len(mfcc[0]), num=len(mfcc[0])), mfcc[0].numpy())
            fig.savefig('graph.png')
        
        return mfcc
    
pipeline = Pipeline(debug=True)

train_target = AudioDataset('data/target_train', transform=pipeline)
train_target_loader = DataLoader(train_target, batch_size=1, shuffle=False)

for i, train_batch in enumerate(train_target_loader):
    print(train_batch["data"][0])