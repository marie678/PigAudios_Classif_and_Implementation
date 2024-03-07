import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()

def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data.unsqueeze(dim=0))
        pred = output.argmax(dim=1, keepdim=True)
        predicted_label = pred.item()
    return predicted_label


def compute_energy(signal):
    squared_signal = torch.square(signal)
    return torch.sum(squared_signal).item()