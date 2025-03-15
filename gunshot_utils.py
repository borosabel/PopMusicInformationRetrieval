import os
import re
import ast
import random
import json
import numpy as np
import pandas as pd
import torch as th
import torchaudio
import librosa
from tqdm import tqdm
import sounddevice as sd
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from pydub.playback import play
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, f1_score, auc
)
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------------------------------------
#                GLOBAL CONSTANTS
# -------------------------------------------------------
SAMPLING_RATE = 44100
HOP_LENGTH = 512
WIN_LENGTHS = [1024, 2048, 4096]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
NUM_FRAMES = 86
FRAME_LENGTH = HOP_LENGTH * (NUM_FRAMES - 1)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
use_cuda = th.cuda.is_available()


# -------------------------------------------------------
#         AUDIO PREPROCESSING & SPECTROGRAMS
# -------------------------------------------------------
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_spectrogram_channels(spectrogram, save=False, filename="spectrogram.png"):
    """
    Plot each channel of the spectrogram individually.
    Optionally save the resulting figure when 'save' is True, and set the file name.

    Args:
        spectrogram (torch.Tensor): A spectrogram tensor of shape [3, 80, 43].
        save (bool): Flag to save the plot. Default is False.
        filename (str): Name of the file when saving the plot. Default is "spectrogram.png".
    """
    spectrogram_np = spectrogram.numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        axs[i].imshow(spectrogram_np[i], aspect='auto', origin='lower', cmap='viridis')
        axs[i].set_title(f'Channel {i + 1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()

    if save:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

def calculate_melbands(waveform, sample_rate):
    """
    Calculate a stack of Mel-spectrograms using different FFT window sizes.
    """
    mel_specs = []
    for wl in WIN_LENGTHS:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=wl,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )(waveform)
        # Convert amplitude to log scale
        mel_specs.append(th.log10(mel_spectrogram + 1e-8))

    # Shape = [3, n_mels, time]
    return th.stack(mel_specs)

def preprocess_audio_train(waveform, label, sample_rate=SAMPLING_RATE):
    """
    Convert the audio waveform into mel spectrogram(s) for model training.
    Returns a list of spectrogram tensors and a list of corresponding labels.
    """
    spectrograms = []
    labels = []
    # If stereo, take the first channel for mel-spectrogram
    mel_specgram = calculate_melbands(waveform[0], sample_rate)
    spectrograms.append(mel_specgram)
    labels.append(label)
    return spectrograms, labels


# -------------------------------------------------------
#      LOADING & SELECTING AUDIO SEGMENTS
# -------------------------------------------------------
def select_gunshot_segment(waveform, sample_rate, start_time, frame_count=FRAME_LENGTH):
    """
    Selects a gunshot segment of length `frame_count` starting at `start_time` in seconds.
    """
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + frame_count

    # Ensure the segment does not exceed the waveform length
    end_sample = min(end_sample, waveform.shape[1])
    start_sample = max(0, end_sample - frame_count)

    return waveform[:, start_sample:end_sample]


def select_valid_onset_segment(file_path, metadata, onset_times, frame_length=FRAME_LENGTH):
    """
    Select a segment starting from a random valid onset time, ensuring the entire
    segment of length `frame_length` fits within the audio duration.
    """
    if not onset_times:
        raise ValueError("The onset_times list is empty.")

    sample_rate = metadata["sample_rate"]
    total_frames = metadata["num_frames"]

    # Filter onsets that fit the required frame length
    valid_onsets = [
        onset for onset in onset_times
        if int(onset * sample_rate) + frame_length <= total_frames
    ]
    if not valid_onsets:
        raise ValueError("No valid onsets found that can accommodate the required frame length.")

    # Pick one random onset
    selected_onset = random.choice(valid_onsets)
    onset_frame = int(selected_onset * sample_rate)

    # Load the exact segment
    waveform, _ = torchaudio.load(
        file_path,
        frame_offset=onset_frame,
        num_frames=frame_length
    )
    return waveform


# -------------------------------------------------------
#         OPTIONAL: AUDIO VISUALIZATION & PLAYBACK
# -------------------------------------------------------
def plot_waveform(waveform, sample_rate=SAMPLING_RATE, vertical_lines=None):
    """
    Plot a waveform with optional vertical lines for reference times.
    """
    if hasattr(waveform, 'numpy'):
        waveform = waveform.numpy()
    if waveform.ndim == 2:
        waveform = waveform[0]

    time_axis = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=time_axis, y=waveform, color='blue')

    if vertical_lines:
        for line_time in vertical_lines:
            plt.axvline(x=line_time, color='red', linestyle='--', linewidth=1)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid(True)
    plt.show()


def play_audio(waveform, sample_rate=SAMPLING_RATE):
    """
    Play an audio waveform using sounddevice.
    """
    try:
        if waveform.ndim == 2:
            num_channels, _ = waveform.shape
            if num_channels == 2:
                waveform_np = waveform.T.numpy()
            else:
                waveform_np = waveform.squeeze().numpy()
        else:
            waveform_np = waveform.numpy()

        sd.play(waveform_np, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error while playing audio: {e}")


# -------------------------------------------------------
#      TRAINING & EVALUATION (If You Need Them)
# -------------------------------------------------------
def compute_mean_std(dataloader):
    l = []
    for features, _, _ in tqdm(dataloader, desc="Computing mean and std"):
        l += features
    tmp = th.cat(l)
    mean = th.mean(tmp, dim=(0, 2)).unsqueeze(1)
    std = th.std(tmp, dim=(0, 2)).unsqueeze(1)
    return mean, std

def train_model(
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        num_epochs=10,
        mean=None,
        std=None,
        patience=3,
        eval_metric='f1',
        device='cuda' if th.cuda.is_available() else 'cpu'
):
    """
    Example training loop with early stopping and learning rate scheduler.
    """
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided for normalization.")

    mean, std = mean.to(device), std.to(device)
    model = model.to(device)
    best_score = 0.0
    best_threshold = 0.5
    epochs_since_improvement = 0

    train_losses = []
    valid_losses = []
    roc_aucs = []
    pr_aucs = []
    last_failed_samples = []

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, verbose=True
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}] Training")
        for features, labels, waveform in train_loader_tqdm:
            features, labels = features.to(device), labels.to(device).float()
            optimizer.zero_grad()
            features = (features - mean) / std

            outputs = model(features).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        valid_loss = 0.0
        with th.no_grad():
            for features, labels, _ in valid_loader:
                features, labels = features.to(device), labels.to(device).float()
                features = (features - mean) / std
                outputs = model(features).view(-1)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * features.size(0)

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        print(f"Epoch [{epoch + 1}], Validation Loss: {valid_loss:.4f}")

        # Evaluate
        eval_score, opt_threshold, roc_auc, pr_auc, failed_samples = evaluate_model_simple(
            model, valid_loader, mean, std, metric=eval_metric
        )
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        last_failed_samples = failed_samples

        print(f"Epoch [{epoch + 1}], AUC-ROC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        if eval_score > best_score:
            best_score = eval_score
            best_threshold = opt_threshold
            epochs_since_improvement = 0
            print(f"New best {eval_metric}: {best_score:.4f} (model saved).")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No improvement in {eval_metric} for {patience} epochs. Stopping training.")
                break

        scheduler.step(eval_score)

    # Confusion matrix after final epoch
    final_confusion_matrix = compute_confusion_matrix(model, valid_loader, best_threshold, mean, std)
    display_confusion_matrix(final_confusion_matrix)

    # Optionally plot training/validation curves here...
    return best_threshold, best_score, last_failed_samples


def evaluate_model_simple(model, valid_loader, mean, std, metric='f1', threshold=0.5,
                          device='cuda' if th.cuda.is_available() else 'cpu'):
    """
    Compute a simple evaluation metric (e.g., F1) along with ROC-AUC and PR-AUC.
    """
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    failed_samples = []

    with th.no_grad():
        for batch_idx, (features, labels, waveform) in enumerate(valid_loader):
            features, labels = features.to(device), labels.to(device).float()
            features = (features - mean) / std
            outputs = model(features).view(-1)

            predictions = (outputs >= threshold).float()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_scores.extend(outputs.cpu().tolist())

            # Track samples the model got wrong
            for i, (lbl, pred, wav) in enumerate(zip(labels, predictions, waveform)):
                if lbl != pred:
                    failed_samples.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'label': lbl.item(),
                        'prediction': pred.item(),
                    })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    if metric == 'f1':
        eval_score = f1_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return eval_score, threshold, roc_auc, pr_auc, failed_samples


def compute_confusion_matrix(model, valid_loader, threshold, mean, std):
    """
    Compute the confusion matrix for final validation.
    """
    all_outputs, all_labels = [], []
    model.eval()

    with th.no_grad():
        for features, labels, _ in tqdm(valid_loader, desc="Computing Confusion Matrix"):
            features = features.to(device)
            labels = labels.cpu().numpy()
            features = (features - mean) / std

            outputs = model(features).view(-1).cpu().numpy()
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    predictions = (all_outputs >= threshold).astype(int)
    return confusion_matrix(all_labels, predictions)


def generate_audio_metadata_from_df(df, filename_col):
    """
    Generates metadata for audio files specified in a DataFrame.

    Parameters:
    - df: pandas.DataFrame, the DataFrame containing file names.
    - filename_col: str, the name of the column in the DataFrame with file paths.

    Returns:
    - metadata_dict: dict, a dictionary with file paths as keys and metadata as values.
    """
    import librosa
    import soundfile as sf
    import os

    # Initialize an empty dictionary to store the metadata
    metadata_dict = {}

    # Iterate through each row in the DataFrame
    for idx, row in df.iterrows():
        file_path = row[filename_col]

        try:
            # Load audio file with librosa to obtain sample rate and duration
            y, sr = librosa.load(file_path, sr=None, mono=False)

            # Read audio metadata with soundfile (e.g., number of channels)
            with sf.SoundFile(file_path) as sound_file:
                channels = sound_file.channels
                num_frames = len(y[0]) if channels > 1 else len(y)
                duration = sound_file.frames / sound_file.samplerate

            # Store metadata in dictionary
            metadata_dict[file_path] = {
                'sample_rate': sr,
                'channels': channels,
                'duration': duration,
                'num_frames': num_frames,
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return metadata_dict


def display_confusion_matrix(cm):
    """
    Utility to display a confusion matrix with [0,1] labels.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='magma')
    plt.title('Confusion Matrix')
    plt.show()
