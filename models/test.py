import os
import torch
import torchaudio
import numpy as np
import pickle as pkl

from torch.utils.data import Dataset, ConcatDataset, Subset
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from glob import glob

torch.manual_seed(0)

class AudioDataset(Dataset):
    # Load audio files from given directory and return them as torchaudio tensor
    def __init__(self, root):
        self.root = root
        self.files = glob(root + '/*.wav')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = torchaudio.load(self.files[idx])
        return {"name": self.files[idx], "data": audio}

class Effect():
    # Apply effect with given probability, parameters is a list of parameter ranges appended to the effect
    def __init__(self, p, effect, parameters=[]):
        self.prob = p
        self.effect = effect
        self.parameters = parameters

    def generate_parameters(self):
        # Randomly generate parameters from specified range and append them to the effect
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
            return torchaudio.sox_effects.apply_effects_tensor(waveform, rate, [effect.split(" ")])[0] # Discard sample rate (does not change)
        return waveform

class Pipeline():
    def __init__(self, in_freq=16000, out_freq=16000, n_fft=256, win_length=200, win_hop=100, n_mels=23, n_mfcc=13, trim=True, effects=[], debug=False):
        # win_length = number of samples in each window - for 200 samples you get 12.5ms windows at 16kHz
        # win_hop = number of samples between the start of two consecutive windows
        # n_fft = number of samples in each window (at 256 with win_length 200 you get 56 zero values at the end of each window)
        # n_mels = number of mel filterbanks
        # n_mfcc = number of mfcc coefficients
        # effects = list of effects to apply to the audio
        self.debug = debug
        self.debug_i = 0

        self.effects = effects
        self.out_freq = out_freq

        self.resample = torchaudio.transforms.Resample(orig_freq=in_freq, new_freq=out_freq)
        # Trim 1.8 seconds from the beginning and silence parts shorter than 0.5 seconds with 0.2% threshold from the end
        self.trim = trim
        self.trim_params = [['trim', '1.8'], ['reverse'], ['silence', '1', '0.3', '0.3%'], ['reverse']]
        melargs = {"n_fft": n_fft, "n_mels": n_mels, "hop_length": win_hop, "win_length": win_length}
        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc, sample_rate=out_freq, melkwargs=melargs)

    def apply(self, x):
        waveform = x
        # Resample the audio
        resampled = self.resample(waveform)
        sample_rate = self.out_freq
        # Trim silent parts from the beginning and the end of the audio
        trimmed = resampled
        if self.trim:
            trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(resampled, sample_rate, self.trim_params)
            if (trimmed.shape[1] < 1):
                trimmed, _ = torchaudio.sox_effects.apply_effects_tensor(resampled, sample_rate, [self.trim_params[0]])
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

def read_dataset(dataset, pipeline, train=True):
    # Read dataset and apply pipeline
    # if `training`` return stacked samples (without name), otherwise return list of (name, sample)
    samples = []
    for sample in dataset:
        sample_data = pipeline.apply(sample["data"][0])[0].numpy().T
        if train:
            samples.append(sample_data)
        else:
            samples.append((sample["name"], sample_data))
    if train:
        return np.vstack(samples)
    return samples

def load_gmm(path='audio_classifier.pkl'):
    GMM_file = open(path, "rb")
    t_gmm, nt_gmm = pkl.load(GMM_file)
    GMM_file.close()
    return t_gmm, nt_gmm

def predict(t_gmm, nt_gmm, data):
    res = []
    for sample in data:
        name = sample[0]
        sample_data = sample[1]
        # P(t) - P(nt) > 0 => target
        score = sum(t_gmm.score_samples(sample_data)) - sum(nt_gmm.score_samples(sample_data))
        res.append({"name": name, "score": score, "target": int(score > 0)})
    return res

def test_pipeline(pipeline, t_data, nt_data):
    val_t = read_dataset(t_data, pipeline, False)
    val_nt = read_dataset(nt_data, pipeline, False)
    target_results = predict(t_gmm, nt_gmm, val_t)
    non_target_results = predict(t_gmm, nt_gmm, val_nt)
    target_correct = sum([1 for r in target_results if r["target"] == 1]) / len(target_results)
    non_target_correct = sum([1 for r in non_target_results if r["target"] == 0]) / len(non_target_results)
    print("Correct target predictions: ", target_correct)
    print("Correct non-target predictions: ", non_target_correct)
    return target_correct, non_target_correct

if __name__ == "__main__":
    # Effects used for augmentation - see `man sox`
    aug_effects = [
        Effect(p=0.4, effect="lowpass -1", parameters=[[200, 1000]]),
        Effect(p=0.4, effect="highpass", parameters=[[800, 3000]]),
        Effect(p=0.7, effect="tempo", parameters=[[0.7, 1.3]]),
        Effect(p=0.2, effect="reverb", parameters=[[20, 50], [20, 50], [20, 40]]),
        Effect(p=0.6, effect="gain", parameters=[[-5, 10]]),
    ]

    aug_effects2 = [
        Effect(p=0.6, effect="lowpass -1", parameters=[[100, 400]]),
        Effect(p=0.6, effect="highpass", parameters=[[1200, 5000]]),
        Effect(p=0.8, effect="tempo", parameters=[[0.5, 2]]),
        Effect(p=0.3, effect="reverb", parameters=[[20, 70], [20, 70], [20, 70]]),
        Effect(p=0.7, effect="gain", parameters=[[-5, 20]]),
    ]

    n_splits = 10     # number of splits to do in stratified shuffle split
    train_size = 0.8 # train size

    print("Reading datasets")
    train_t = AudioDataset('../data/train/target_train')
    val_t = AudioDataset('../data/val/target_dev')
    train_nt = AudioDataset('../data/train/non_target_train')
    val_nt = AudioDataset('../data/val/non_target_dev')

    dataset = ConcatDataset([train_t, val_t, train_nt, val_nt])
    labels = np.concatenate([np.array([1] *(len(train_t) + len(val_t))), np.array([0] * (len(train_nt) + len(val_nt)))])
    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size)

    overall_acc = [0, 0, 0, 0, 0, 0] # default correct target, default correct non-target, augmented correct target, augmented correct non-target
    for i, (train_index, test_index) in enumerate(sss.split(dataset, labels)):
        print(f"\nBatch {i}")
        train_t_idx = train_index[np.argwhere(labels[train_index] == 1)].ravel()
        train_nt_idx = train_index[np.argwhere(labels[train_index] == 0)].ravel()
        t_gmm, nt_gmm = load_gmm('../audio_classifier.pkl')

        train_t = Subset(dataset, train_t_idx)
        train_nt = Subset(dataset, train_nt_idx)

        # Apply the pipeline without effects to the validation data and test the model
        default_pipeline = Pipeline()
        print("Default data")
        acc_def_t, acc_def_nt = test_pipeline(default_pipeline, val_t, val_nt)

        # Apply the pipeline with basic effects (training) to the validation data and test the model
        aug_pipeline = Pipeline(effects=aug_effects)
        print("Augmented data")
        acc_aug_t, acc_aug_nt = test_pipeline(aug_pipeline, val_t, val_nt)

        # Apply the pipeline with basic effects (training) to the validation data and test the model
        aug_pipeline2 = Pipeline(effects=aug_effects2)
        print("More augmented data")
        acc_maug_t, acc_maug_nt = test_pipeline(aug_pipeline2, val_t, val_nt)

        overall_acc[0] += acc_def_t
        overall_acc[1] += acc_def_nt
        overall_acc[2] += acc_aug_t
        overall_acc[3] += acc_aug_nt
        overall_acc[4] += acc_maug_t
        overall_acc[5] += acc_maug_nt

    print("\nOverall")
    print(f"Default target accuracy: {overall_acc[0] / n_splits}, non-target accuracy: {overall_acc[1] / n_splits}")
    print(f"Augmented target accuracy: {overall_acc[2] / n_splits}, non-target accuracy: {overall_acc[3] / n_splits}")
    print(f"More Augmented target accuracy: {overall_acc[4] / n_splits}, non-target accuracy: {overall_acc[5] / n_splits}")

