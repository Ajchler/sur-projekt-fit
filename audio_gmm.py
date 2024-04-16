import torch
import torchaudio
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset
from glob import glob

torch.manual_seed(0)

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
    
class Effect():
    # Apply effect with given probability, parameters is a list of parameter ranges for the effect
    def __init__(self, p, effect, parameters=[]):
        self.effect = effect
        self.parameters = parameters
        self.prob = p
        
    def generate_parameters(self):
        effect = self.effect
        for i in range(len(self.parameters)):
            if type(self.parameters[i][0]) == int:
                effect += " " + str(torch.randint(self.parameters[i][0], self.parameters[i][1], (1,)).item())
            elif type(self.parameters[i][0]) == float:
                effect += " " + str(torch.rand(1).item() * (self.parameters[i][1] - self.parameters[i][0]) + self.parameters[i][0])
            else:
                raise ValueError("Parameter type not recognized")
        return effect
    
    def apply(self, waveform, rate):
        effect = self.effect
        if torch.rand(1) < self.prob:
            if len(self.parameters) > 0:
                effect = self.generate_parameters()
            return torchaudio.sox_effects.apply_effects_tensor(waveform, rate, [effect.split(" ")])[0]
        return waveform

class Pipeline(torch.nn.Module):
    def __init__(self, in_freq=16000, out_freq=16000, n_fft=256, win_length=200, win_hop=100, n_mels=23, n_mfcc=13, debug=False):
        # win_length = number of samples in each window - for 200 samples you get 12.5ms windows at 16kHz
        # win_hop = number of samples between the start of two consecutive windows
        # n_fft = number of samples in each window (at 256 with win_length 200 you get 56 zero values at the end of each window)
        # n_mels = number of mel filterbanks
        # n_mfcc = number of mfcc coefficients
        super(Pipeline, self).__init__()
        
        self.debug = debug
        self.debug_i = 0
        self.out_freq = out_freq
        
        self.resample = torchaudio.transforms.Resample(orig_freq=in_freq, new_freq=out_freq)
        # Trim 1.8 seconds from the beginning and silence parts shorter than 0.5 seconds with 0.2% threshold from the end
        self.trim_params = [['trim', '1.8'], ['reverse'], ['silence', '1', '0.5', '0.2%'], ['reverse']]
        
        self.effects = [
            # For more effects see man sox
            # Effect(p=0.2, effect="lowpass -1", parameters=[[200, 1000]]),
            # Effect(p=0.2, effect="highpass", parameters=[[800, 3000]]),
            # Effect(p=0.2, effect="tempo", parameters=[[0.7, 1.3]]),
            # Effect(p=0.2, effect="reverb", parameters=[[20, 50], [20, 50], [20, 40]]),
            # Effect(p=0.4, effect="gain", parameters=[[-5, 10]]),
        ]
        
        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, sample_rate=out_freq, log_mels=True, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": win_hop, "win_length": win_length})
        
    def forward(self, x):
        waveform, sample_rate = x
        
        # Resample the audio
        resampled = self.resample(waveform)
        sample_rate = self.out_freq
        # Trim silent parts from the beginning and the end of the audio
        trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(resampled, sample_rate, self.trim_params)
        # Apply effects
        augmented = trimmed
        for effect in self.effects:
            augmented = effect.apply(augmented, sample_rate)
        if self.debug:
            torchaudio.save('augmented/' + str(self.debug_i) + '.wav', augmented, sample_rate)
            self.debug_i += 1
        # Get MFCC coefficients
        mfcc = self.mfcc(augmented)
        
        return mfcc
    
def read_dataset(dataset, stack=True):
    samples = []
    for sample in dataset:
        samples.append(sample["data"][0].numpy().T)
    if stack:
        return np.vstack(samples)
    return samples
    
pipeline = Pipeline()

print("Reading datasets")
train_t = read_dataset(dataset=AudioDataset('data/target_train', transform=pipeline))
train_nt = read_dataset(dataset=AudioDataset('data/non_target_train', transform=pipeline))
val_t = read_dataset(dataset=AudioDataset('data/target_dev', transform=pipeline), stack=False)
val_nt = read_dataset(dataset=AudioDataset('data/non_target_dev', transform=pipeline), stack=False)

print(f"Training target shape: {train_t.shape}")
print(f"Traning non-target shape: {train_nt.shape}")

t_gmm = GaussianMixture(n_components=4, verbose=2)
nt_gmm = GaussianMixture(n_components=20, verbose=2)
print("Fitting target GMM")
t_gmm.fit(train_t)
print("Fitting non-target GMM")
nt_gmm.fit(train_nt)

tscores = []
for sample in val_t:
    tscore = sum(t_gmm.score_samples(sample))
    ntscore = sum(nt_gmm.score_samples(sample))
    tscores.append(tscore - ntscore)
    
print("Correcly classified target samples: ", sum([1 for score in tscores if score > 0]) / len(tscores))

ntscores = []
for sample in val_nt:
    tscore = sum(t_gmm.score_samples(sample))
    ntscore = sum(nt_gmm.score_samples(sample))
    ntscores.append(tscore - ntscore)
    
print("Correcly classified non-target samples: ", sum([1 for score in ntscores if score < 0]) / len(ntscores))