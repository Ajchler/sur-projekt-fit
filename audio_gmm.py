import torch
import torchaudio
import numpy as np

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
    # Apply effect with given probability, parameters is a list of parameter ranges appended to the effect
    def __init__(self, p, effect, parameters=[]):
        self.prob = p
        self.effect = effect
        self.parameters = parameters
        
    def generate_parameters(self):
        effect = self.effect
        for i in range(len(self.parameters)):
            # Randomly generate parameters from specified range
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
            return torchaudio.sox_effects.apply_effects_tensor(waveform, rate, [effect.split(" ")])[0] # Discard sample rate (does not change)
        return waveform

class Pipeline(torch.nn.Module):
    def __init__(self, in_freq=16000, out_freq=16000, n_fft=256, win_length=200, win_hop=100, n_mels=23, n_mfcc=13, effects=[], debug=False):
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
        self.effects = effects
        melargs = {"n_fft": n_fft, "n_mels": n_mels, "hop_length": win_hop, "win_length": win_length}
        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, sample_rate=out_freq, melkwargs=melargs)
        
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

def read_dataset_n(dataset, n=5):
    # Read dataset N times (apply different augmentations)
    res = []
    for _ in range(n):
        res.append(read_dataset(dataset=dataset))
    return np.vstack(res)

# Effects used for augmentation - see man sox
aug_effects = [
    Effect(p=0.4, effect="lowpass -1", parameters=[[200, 1000]]),
    Effect(p=0.4, effect="highpass", parameters=[[800, 3000]]),
    Effect(p=0.7, effect="tempo", parameters=[[0.7, 1.3]]),
    Effect(p=0.3, effect="reverb", parameters=[[20, 50], [20, 50], [20, 40]]),
    Effect(p=0.7, effect="gain", parameters=[[-5, 10]]),
]

# First pipeline is used for training (augmentation), second for validation
aug_pipeline = Pipeline(effects=aug_effects)
val_pipeline = Pipeline()

print("Reading datasets")
train_t = read_dataset_n(dataset=AudioDataset('data/target_train', transform=aug_pipeline))
train_nt = read_dataset_n(dataset=AudioDataset('data/non_target_train', transform=aug_pipeline))

val_t = read_dataset(dataset=AudioDataset('data/target_dev', transform=val_pipeline), stack=False)
val_nt = read_dataset(dataset=AudioDataset('data/non_target_dev', transform=val_pipeline), stack=False)

print(f"Training target shape: {train_t.shape}")
print(f"Traning non-target shape: {train_nt.shape}")

t_gmm = GaussianMixture(n_components=7, verbose=2)
nt_gmm = GaussianMixture(n_components=20, verbose=2)
print("\nFitting target GMM")
t_gmm.fit(train_t)
print("\nFitting non-target GMM")
nt_gmm.fit(train_nt)

print()

tscores = []
for sample in val_t:
    tscore = sum(t_gmm.score_samples(sample))
    ntscore = sum(nt_gmm.score_samples(sample))
    tscores.append(tscore - ntscore)

print(tscores)
print("Correcly classified target samples: ", sum([1 for score in tscores if score > 0]) / len(tscores))

ntscores = []
for sample in val_nt:
    tscore = sum(t_gmm.score_samples(sample))
    ntscore = sum(nt_gmm.score_samples(sample))
    ntscores.append(tscore - ntscore)
    
print(ntscores)
print("Correcly classified non-target samples: ", sum([1 for score in ntscores if score < 0]) / len(ntscores))