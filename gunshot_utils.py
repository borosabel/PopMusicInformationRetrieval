import random
import numpy as np
import torchaudio
import torch as th
from tqdm import tqdm
import librosa
from pydub import AudioSegment
import re
import ast
import os
import sounddevice as sd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pydub.playback import play
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score,
    recall_score, f1_score, average_precision_score,
)

SAMPLING_RATE = 44100
HOP_LENGTH = 512
WIN_LENGTHS = [1024, 2048, 4096]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
NUM_FRAMES = 86
FRAME_LENGTH = HOP_LENGTH * (NUM_FRAMES - 1)


def preprocess_gunshot_times(gunshot_times, include_first_gunshot_only=False):
    """Preprocess the gunshot timing data."""
    if not isinstance(gunshot_times, str):
        return []
    gunshot_times = re.sub(r'\s+', ' ', gunshot_times).strip()
    gunshot_times = re.sub(r'(?<=\d)\s(?=\d)', ', ', gunshot_times)
    gunshot_times = gunshot_times.replace(', ]', ']')
    try:
        gunshot_list = ast.literal_eval(gunshot_times)
        if not isinstance(gunshot_list, list):
            return []
        gunshot_list = [float(x) for x in gunshot_list if isinstance(x, (int, float))]
        if include_first_gunshot_only and gunshot_list:
            return [gunshot_list[0]]
        return gunshot_list
    except (ValueError, SyntaxError):
        return []


def plot_spectrogram_channels(spectrogram):
    """
    Plot each channel of the spectrogram individually.
    Args:
        spectrogram (torch.Tensor): A spectrogram tensor of shape [3, 80, 43].
    """
    spectrogram_np = spectrogram.numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        axs[i].imshow(spectrogram_np[i], aspect='auto', origin='lower', cmap='viridis')
        axs[i].set_title(f'Channel {i + 1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_spectrogram_channels_two_rows(spectogram_onset, spectogram_gunshot):
    """
    Plot each channel of two spectrograms individually in a 2-row by 3-column layout.

    Args:
        spectrogram_1 (torch.Tensor): The first spectrogram tensor
        spectrogram_2 (torch.Tensor): The second spectrogram tensor
    """
    # Convert both spectrogram tensors to numpy arrays
    spectrogram_1_np = spectogram_onset.numpy()
    spectrogram_2_np = spectogram_gunshot.numpy()

    # Create a 2-row by 3-column figure for plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the first spectrogram's channels in the first row
    for i in range(3):
        axs[0, i].imshow(spectrogram_1_np[i], aspect='auto', origin='lower', cmap='viridis')
        axs[0, i].set_title(f'Spectogram Onset - Channel {i + 1}')
        axs[0, i].set_xlabel('Time')
        axs[0, i].set_ylabel('Frequency')

    # Plot the second spectrogram's channels in the second row
    for i in range(3):
        axs[1, i].imshow(spectrogram_2_np[i], aspect='auto', origin='lower', cmap='viridis')
        axs[1, i].set_title(f'Spectogram Gunshot - Channel {i + 1}')
        axs[1, i].set_xlabel('Time')
        axs[1, i].set_ylabel('Frequency')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_spectrogram_rgb(spectrogram):
    """
    Plot the spectrogram as an RGB image.
    Args:
        spectrogram (torch.Tensor): A spectrogram tensor of shape [3, 80, 43].
    """
    # Convert to numpy and adjust shape to be compatible with matplotlib
    spectrogram_np = spectrogram.numpy()
    spectrogram_rgb = np.transpose(spectrogram_np, (1, 2, 0))  # Convert to shape [80, 43, 3]

    plt.figure(figsize=(8, 5))
    plt.imshow(spectrogram_rgb, aspect='auto', origin='lower')
    plt.title('RGB Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()


def plot_waveform(waveform, sample_rate=SAMPLING_RATE, vertical_lines=None):
    """
    Plot an audio waveform with optional vertical lines.

    Parameters:
    waveform (Tensor or ndarray): The audio waveform.
    sample_rate (int): The sample rate of the audio.
    vertical_lines (list of float, optional): List of times (in seconds) where vertical lines should be plotted.
    """
    try:
        # Convert waveform to numpy if needed
        if hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()

        # If stereo, select one channel for plotting
        if waveform.ndim == 2:
            waveform = waveform[0]

        # Create time axis in seconds
        time_axis = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))

        # Plot the waveform
        plt.figure(figsize=(10, 4))
        sns.lineplot(x=time_axis, y=waveform, color='blue')

        # Plot vertical lines if specified
        if vertical_lines:
            for line_time in vertical_lines:
                plt.axvline(x=line_time, color='red', linestyle='--', linewidth=1)

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.grid(True)
        plt.show(block=False)
    except Exception as e:
        print(f"Error occurred while plotting waveform: {e}")


def play_audio(waveform, sample_rate=SAMPLING_RATE):
    """
    Play an already loaded audio waveform using sounddevice.

    Parameters:
    waveform (Tensor or ndarray): The audio waveform.
    sample_rate (int): The sample rate of the audio.
    """
    try:
        if waveform.ndim == 2:
            num_channels, num_samples = waveform.shape

            if num_channels == 2:
                waveform_np = waveform.T.numpy()
            elif num_channels == 1:
                waveform_np = waveform.squeeze().numpy()
            else:
                raise ValueError(f"Unsupported number of channels: {num_channels}")
        else:
            waveform_np = waveform.numpy()

        # Play the audio using sounddevice
        sd.play(waveform_np, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error occurred while playing audio: {e}")


def resample_and_convert_to_wav(df, source_column, target_folder, target_sample_rate=44100, limit=None):
    """
    Resample all audio files in the DataFrame to the target sample rate and save them as WAV in the target folder.

    :param df: DataFrame containing the file paths to the original audio files.
    :param source_column: The column in the DataFrame that contains the file paths.
    :param target_folder: Folder where the resampled WAV files will be saved.
    :param target_sample_rate: The sample rate to which all audio should be resampled.
    :param limit: Optional. The maximum number of audio files to process.
    :return: DataFrame with updated file paths, sample rates, and file extensions.
    """

    # Ensure the target directory exists
    os.makedirs(target_folder, exist_ok=True)

    # Apply the limit if provided
    if limit:
        df = df.head(limit)

    # Lists to store the updated file paths and sample rates
    updated_paths = []
    updated_sample_rates = []

    # Iterate over all file paths in the DataFrame with tqdm progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing audio files"):
        source_file_path = row[source_column]
        file_name = os.path.basename(source_file_path)
        target_file_path = os.path.join(target_folder, os.path.splitext(file_name)[0] + '.wav')

        try:
            # Load the audio file
            audio_waveform, original_sample_rate = torchaudio.load(source_file_path)

            # Resample if necessary
            if original_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate,
                                                           new_freq=target_sample_rate)
                audio_waveform = resampler(audio_waveform)
                updated_sample_rate = target_sample_rate
            else:
                updated_sample_rate = original_sample_rate

            # Save the resampled audio file as WAV in the target folder
            torchaudio.save(target_file_path, audio_waveform, target_sample_rate)

            # Append the new file path and sample rate to the lists
            updated_paths.append(target_file_path)
            updated_sample_rates.append(updated_sample_rate)

        except Exception as e:
            tqdm.write(f"Error processing {file_name}: {e}")
            updated_paths.append(None)  # In case of error, append None for that file
            updated_sample_rates.append(None)  # In case of error, append None for the sample rate

    df['Path'] = updated_paths
    df['Sample Rate (Hz)'] = updated_sample_rates
    df = df.dropna(subset=['Path'])

    df['file_extension'] = '.wav'

    return df


def create_metadata_map(dataset_path, save_metadata=False):
    """
    Creates a metadata map for WAV audio files using torchaudio.

    :param dataset_path: Path to the folder containing WAV audio files.
    :param save_metadata: Boolean indicating whether to save metadata to a JSON file.
    :return: A dictionary containing metadata for each WAV file in the dataset.
    """
    metadata_map = {}

    # Iterate over all files in the dataset directory
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):  # Only process WAV files
            file_path = os.path.join(dataset_path, file_name)

            try:
                # Use torchaudio.info() to get file metadata
                info = torchaudio.info(file_path)

                # Calculate the duration in seconds
                duration = info.num_frames / info.sample_rate

                # Store metadata in the dictionary
                metadata_map[file_path] = {
                    "sample_rate": info.sample_rate,
                    "channels": info.num_channels,
                    "duration": duration,  # Duration in seconds
                    "num_frames": info.num_frames
                }

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue

    # Save metadata map to a JSON file if save_metadata is True
    if save_metadata:
        with open('metadata_map_torchaudio.json', 'w') as json_file:
            json.dump(metadata_map, json_file)

    return metadata_map


def extract_music_segment(music_file, excerpt_len=5.0, sample_rate=44100):
    """
    Extracts a segment from a music file of the specified length.

    :param music_file: Path to the music file.
    :param excerpt_len: Length of the music segment to extract (in seconds).
    :param sample_rate: Sample rate to process the audio.

    :return: Extracted music segment.
    """

    # Load the music file
    music_waveform, sr = torchaudio.load(music_file)
    if sr != sample_rate:
        music_waveform = torchaudio.transforms.Resample(sr, sample_rate)(music_waveform)

    excerpt_len_samples = int(excerpt_len * sample_rate)

    # Ensure the music segment is within bounds
    total_music_samples = music_waveform.size(1)
    max_start_sample = max(0, total_music_samples - excerpt_len_samples)
    start_pos_music = random.randint(0, max_start_sample)
    music_segment = music_waveform[:, start_pos_music:start_pos_music + excerpt_len_samples]

    return music_segment, sample_rate


def combine_music_and_gunshot(music_waveform, gunshot_file, gunshot_time, gunshot_volume_increase_dB=8, sample_rate=44100, pre_gunshot_time=0):
    """
    Combines a music segment with a gunshot starting at the beginning of the music waveform.

    :param music_waveform: Preloaded waveform tensor of the music.
    :param gunshot_file: Path to the gunshot audio file.
    :param gunshot_time: Time in seconds where the gunshot occurs in the gunshot file.
    :param gunshot_volume_increase_dB: Volume increase for the gunshot in decibels.
    :param sample_rate: The sample rate of the audio.
    :param pre_gunshot_time: Time in seconds to include before the gunshot.
    :return: Combined music and gunshot waveform, sample rate.
    """

    # --- DEALING WITH THE GUNSHOT FILE ---
    # Load the gunshot file
    gunshot_waveform, _ = torchaudio.load(gunshot_file)

    # Extract the relevant gunshot segment based on gunshot_time and pre_gunshot_time
    if gunshot_time >= pre_gunshot_time:
        gunshot_start_sample = int((gunshot_time - pre_gunshot_time) * sample_rate)
    else:
        gunshot_start_sample = 0

    # Ensure the gunshot segment length matches the music waveform length
    music_length_samples = music_waveform.size(1)
    gunshot_segment = gunshot_waveform[:, gunshot_start_sample:gunshot_start_sample + music_length_samples]

    if gunshot_segment.size(1) < music_length_samples:
        pad_length = music_length_samples - gunshot_segment.size(1)
        gunshot_segment = th.nn.functional.pad(gunshot_segment, (0, pad_length))

    # Apply volume gain to gunshot
    gain_factor = 10 ** (gunshot_volume_increase_dB / 20)
    gunshot_segment *= gain_factor

    # --- DEALING WITH THE MUSIC AND GUNSHOT OVERLAY ---
    # Overlay the gunshot onto the music from the beginning
    combined_segment = music_waveform.clone()
    combined_segment[:, :gunshot_segment.size(1)] += gunshot_segment

    return combined_segment, sample_rate


def preprocess_audio_train(waveform, label, sample_rate=SAMPLING_RATE):
    """
    Preprocess a single audio waveform (either music with or without gunshots) to generate mel spectrograms for model training.

    Parameters:
        waveform (Tensor): The audio waveform (music or music+gunshot).
        sample_rate (int): Sample rate of the waveform.
        label (int): 1 for gunshot, 0 for no gunshot.
        gunshot_time (float, optional): The time of the gunshot in seconds (only for gunshot samples).

    Returns:
        spectrograms (list): List of spectrograms for training.
        labels (list): List of labels corresponding to each spectrogram.
    """
    # print("------PREPROCESSING AUDIO DATA------")
    spectrograms = []
    labels = []
    mel_specgram = calculate_melbands(waveform[0], sample_rate)
    spectrograms.append(mel_specgram)
    labels.append(label)
    # print("------PREPROCESSING AUDIO DATA------")
    return spectrograms, labels


def select_random_segment(file_path, metadata, frame_length=FRAME_LENGTH):
    """
    Select a random segment of fixed frame length from an audio file using torchaudio.

    :param file_path: Path to the audio file.
    :param sample_rate: The sample rate of the audio.
    :param frame_length: The desired number of frames to load (e.g., 44100 for 1 second at 44.1 kHz).
    :param metadata: Metadata dictionary containing "num_frames" for the audio file.
    :return: Loaded waveform containing the random segment of length `frame_length`.
    """

    # Retrieve metadata for the audio file
    total_frames = metadata["num_frames"]

    # If the total number of frames is less than or equal to the frame length, load the whole file
    if total_frames <= frame_length:
        waveform, sr = torchaudio.load(file_path)
        return waveform

    # Calculate a random starting frame position such that the loaded segment remains within bounds
    max_start_frame = total_frames - frame_length
    start_frame = random.randint(0, max_start_frame)

    # Load only the required segment from the audio file
    waveform, sr = torchaudio.load(
        file_path,
        frame_offset=start_frame,
        num_frames=frame_length
    )

    return waveform

def select_valid_onset_segment(file_path, metadata, onset_times, frame_length=FRAME_LENGTH):
    """
    Select a valid segment starting from an onset time in an audio file using torchaudio.
    Only selects an onset where the frame length fits within the audio duration.

    :param file_path: Path to the audio file.
    :param metadata: Metadata dictionary containing "num_frames" and "sample_rate".
    :param onset_times: A list of onset times in seconds.
    :param frame_length: The desired number of frames to load (e.g., 44100 for 1 second at 44.1 kHz).
    :return: Loaded waveform containing the segment starting at the selected onset.
    """
    if len(onset_times) == 0:
        raise ValueError("The onset_times list is empty.")

    sample_rate = metadata["sample_rate"]
    total_frames = metadata["num_frames"]

    # Filter onsets to keep only those that can fit a full segment
    valid_onsets = [onset for onset in onset_times if int(onset * sample_rate) + frame_length <= total_frames]

    # Check if we have any valid onsets left after filtering
    if len(valid_onsets) == 0:
        raise ValueError("No valid onsets found that can accommodate the required frame length.")

    # Randomly select an onset from the valid list
    selected_onset = random.choice(valid_onsets)
    onset_frame = int(selected_onset * sample_rate)

    # Load the segment starting at the selected onset
    waveform, sr = torchaudio.load(
        file_path,
        frame_offset=onset_frame,
        num_frames=frame_length
    )

    return waveform

def process_and_predict(model, audio_path, start_time_sec, mean, std, threshold):

    print(f"Treshold {threshold}")

    # Extract the waveform and the audio sample
    waveform, sample, sample_rate = extract_sample_at_time(audio_path, start_time_sec)

    # utils.plot_waveform(waveform)
    # time.sleep(1)  # Add a small delay to ensure the plot gets time to render

    print(f"Playing the audio sample from {start_time_sec:.2f} seconds.")
    play(sample)

    mean = mean.to(device)
    std = std.to(device)
    model = model.to(device)
    waveform = waveform.to(device)

    mel_spectrogram = calculate_melbands(waveform[0], sample_rate)
    mel_spectrogram = (mel_spectrogram - mean) / std

    # Reshape and feed to model
    with th.no_grad():
        input_tensor = mel_spectrogram.unsqueeze(0).float()
        output = model(input_tensor).squeeze().item()

    if output >= threshold:
        prediction = "Gunshot"
    else:
        prediction = "No Gunshot"
    print(f"Model Prediction: {prediction} with output: {output}")
    return prediction


def frames_to_seconds(frame_count, sample_rate):
    """
    Convert a number of frames to seconds based on the sample rate.

    Parameters:
    frame_count (int): The number of frames.
    sample_rate (int): The sample rate of the audio.

    Returns:
    float: The equivalent time in seconds.
    """
    seconds = frame_count / sample_rate
    print(
        f"Frame count: {frame_count} Number of frames: {frame_count} corresponds to {seconds:.4f} seconds at a sample rate of {sample_rate} Hz.")
    return seconds


def select_gunshot_segment(waveform, sample_rate, start_time, frame_count=FRAME_LENGTH):
    """
    Selects a segment of audio starting from a given time and lasting for a given number of frames.

    Parameters:
        waveform (Tensor or ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        start_time (float): The start time in seconds for selecting the segment.
        frame_count (int): The number of frames to select.

    Returns:
        Tensor or ndarray: The selected segment of audio.
    """
    # frames_to_seconds(frame_count, sample_rate)

    # Calculate start and end sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + frame_count

    # Ensure we don't exceed the total length of the waveform
    end_sample = min(end_sample, waveform.shape[1])
    start_sample = max(0, end_sample - frame_count)

    return waveform[:, start_sample:end_sample]


def calculate_melbands(waveform, sample_rate):
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
        mel_specs.append(mel_spectrogram)

    mel_specs = th.log10(th.stack(mel_specs) + 1e-08)

    return mel_specs


def preprocess_audio(files):
    spectrograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_specgram = calculate_melbands(waveform[0], sample_rate)
        spectrograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectrograms, sample_rates


def compute_mean_std(dataloader):
    l = []
    for features, _ in tqdm(dataloader, desc="Computing mean and std"):
        l += features
    tmp = th.cat(l)
    mean = th.mean(tmp, dim=(0, 2)).unsqueeze(1)
    std = th.std(tmp, dim=(0, 2)).unsqueeze(1)
    return mean, std


def extract_sample_at_time(audio_path, start_time_sec, frame_size=NUM_FRAMES, hop_length=HOP_LENGTH):
    # Load the full audio file
    audio = AudioSegment.from_file(audio_path)

    _, sample_rate = torchaudio.load(audio_path)

    sample_duration_sec = (frame_size - 1) * hop_length / sample_rate
    sample_duration_ms = sample_duration_sec * 1000

    # Calculate start time in milliseconds
    start_time_ms = start_time_sec * 1000

    # Extract the segment using pydub
    sample = audio[start_time_ms:start_time_ms + sample_duration_ms]

    frame_offset = int(start_time_sec * sample_rate)
    num_frames = int(sample_duration_sec * sample_rate)

    waveform, _ = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)

    return waveform, sample, sample_rate


##########################################################################################################################

device = th.device("cuda" if th.cuda.is_available() else "cpu")
use_cuda = th.cuda.is_available()
import itertools

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
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided for normalization.")

    mean, std = mean.to(device), std.to(device)
    model = model.to(device)
    best_score = 0.0
    best_threshold = 0.5
    epochs_since_improvement = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training Phase
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}] Training")
        for features, labels in train_loader_tqdm:
            features, labels = features.to(device), labels.to(device).float()
            optimizer.zero_grad()
            features = (features - mean) / std

            outputs = model(features).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}")

        # Validation Phase
        eval_score, optimal_threshold = evaluate_model_simple(model, valid_loader, mean, std, metric=eval_metric)
        if eval_score > best_score:
            best_score = eval_score
            best_threshold = optimal_threshold
            epochs_since_improvement = 0
            print(f"New best {eval_metric}: {best_score:.4f}, model saved.")
            # You can save the model here if desired
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1
        # Early stopping
        if epochs_since_improvement >= patience:
            print(f"No improvement in {eval_metric} for {patience} epochs. Stopping training.")
            break

    # Final Confusion Matrix and Result
    final_confusion_matrix = compute_confusion_matrix(model, valid_loader, best_threshold, mean, std)
    display_confusion_matrix(final_confusion_matrix)

    return best_threshold, best_score

def evaluate_model_simple(model, valid_loader, mean, std, metric='f1'):
    all_outputs = []
    all_labels = []

    model.eval()
    with th.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Validation")
        for features, labels in valid_loader_tqdm:
            features = features.to(device)
            labels = labels.to(device).float()
            features = (features - mean) / std
            outputs = model(features).view(-1).cpu().numpy()

            all_outputs.append(outputs)
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Compute optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    youdens_j = tpr - fpr
    idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Binarize outputs based on the optimal threshold
    binary_outputs = (all_outputs >= optimal_threshold).astype(int)

    # Calculate the requested metric
    if metric == 'f1':
        value = f1_score(all_labels, binary_outputs)
    elif metric == 'precision':
        value = precision_score(all_labels, binary_outputs)
    elif metric == 'recall':
        value = recall_score(all_labels, binary_outputs)
    elif metric == 'roc_auc':
        value = roc_auc_score(all_labels, all_outputs)
    elif metric == 'average_precision':
        value = average_precision_score(all_labels, all_outputs)
    else:
        raise ValueError(f"Unknown metric: {metric}. Please choose from 'f1', 'precision', 'recall', 'roc_auc', 'average_precision'.")

    return value, optimal_threshold

def compute_confusion_matrix(model, valid_loader, threshold, mean, std):
    all_outputs = []
    all_labels = []

    model.eval()
    with th.no_grad():
        valid_loader_tqdm = tqdm(valid_loader, desc="Computing Confusion Matrix")
        for features, labels in valid_loader_tqdm:
            features = features.to(device)
            labels = labels.cpu().numpy()
            features = (features - mean) / std
            outputs = model(features).view(-1).cpu().numpy()

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    predictions = (all_outputs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, predictions)

    return cm

def display_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='magma')
    plt.title('Confusion Matrix')
    plt.show()


def manual_evaluate_test(model, feature, threshold, frame_size=NUM_FRAMES, sampling_rate=SAMPLING_RATE, hop_length=HOP_LENGTH, mean=None, std=None, step_size=None, filter_time_sec=0):
    """
    Manually evaluate the model on an audio feature, returning time positions where gunshots are detected.

    Parameters:
        model: The trained model.
        feature: The feature (e.g., spectrogram) to evaluate.
        threshold: The prediction threshold for gunshots.
        frame_size: Number of frames to use in each evaluation.
        sampling_rate: Audio sampling rate.
        hop_length: Hop length in samples for each frame.
        mean: Mean for normalization.
        std: Standard deviation for normalization.
        step_size: Step size for moving through frames (default: frame_size // 2).
        filter_time_sec: Time (in seconds) to filter out close consecutive predictions.

    Returns:
        List of tuples (minutes, seconds, output) where gunshots are detected along with the model's output.
    """
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided for normalization.")

    mean = mean.to(device)
    std = std.to(device)
    model = model.to(device)
    model.eval()

    predictions = []

    feature = feature.to(device)
    feature = (feature - mean) / std

    num_frames = feature.shape[2]

    if step_size is None:
        step_size = 1

    total_iterations = 0

    with th.no_grad():
        for j in range(0, num_frames - frame_size + 1, step_size):
            total_iterations += 1
            start = j
            end = j + frame_size

            input_frame = feature[:, :, start:end].unsqueeze(0).float()
            output = model(input_frame).squeeze().item()
            predictions.append((output, start))

        print("Number of predictions", len(predictions))

        res = []
        for output, start in predictions:
            if output >= threshold:
                time_in_seconds = start * hop_length / sampling_rate
                minutes = int(time_in_seconds // 60)
                seconds = time_in_seconds % 60
                res.append((minutes, seconds, time_in_seconds, output))

    print(f'Found {len(res)} gunshot samples.')

    filtered_res = []
    last_detection_time = -float('inf')

    for minutes, seconds, time_in_seconds, output in res:
        if time_in_seconds - last_detection_time >= filter_time_sec:
            filtered_res.append((minutes, seconds, output))
            last_detection_time = time_in_seconds

    return filtered_res

#######################################################################################


def calculate_loudness(audio_path):
    """
    Calculate the maximum loudness of an audio file in decibels.

    :param audio_path: Path to the audio file.
    :return: Maximum loudness in decibels.
    """
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate the RMS (Root Mean Square) energy of the entire audio signal
        rms = librosa.feature.rms(y=y)

        # Convert RMS to decibels using a fixed reference (e.g., ref=1.0)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)

        # Take the maximum loudness value
        max_loudness = rms_db.max()

        return max_loudness

    except Exception as e:
        print(f"Error calculating loudness for {audio_path}: {e}")
        return None

def filter_loud_gunshots(df, filepath_column, loudness_threshold):
    """
    Filters the dataset to include only rows where the gunshot audio file exceeds the given loudness threshold.

    :param df: DataFrame containing the file paths to the audio files.
    :param filepath_column: The column in the DataFrame that contains the file paths.
    :param loudness_threshold: The loudness threshold in decibels.
    :return: Filtered DataFrame containing only rows with loud gunshots.
    """
    # List to store the maximum loudness values
    loudness_values = []

    # Calculate the maximum loudness for each audio file in the DataFrame
    for index, row in df.iterrows():
        audio_path = row[filepath_column]
        try:
            max_loudness = calculate_loudness(audio_path)
            print(max_loudness)
            loudness_values.append(max_loudness)
        except Exception as e:
            # In case of an error, set loudness to None
            print(f"Error processing {audio_path}: {e}")
            loudness_values.append(None)

    # Add the loudness values to the DataFrame
    df['Max Loudness (dB)'] = loudness_values

    # Drop rows where loudness could not be calculated (None values)
    df = df.dropna(subset=['Max Loudness (dB)'])

    # Filter rows where the maximum loudness is greater than or equal to the threshold
    filtered_df = df[df['Max Loudness (dB)'] >= loudness_threshold]

    return filtered_df

def filter_by_firearm(df, gun_types):
    """
    Filters the dataset to include only rows where the firearm contains any of the specified gun types (partial match).

    :param df: The input DataFrame containing the 'firearm' column.
    :param gun_types: A list of gun types to filter for (partial match).
    :return: A filtered DataFrame containing only rows with the specified gun types.
    """
    # Combine all conditions for partial matching
    mask = df['firearm'].str.lower().str.contains(gun_types[0].lower())
    for gun in gun_types[1:]:
        mask |= df['firearm'].str.lower().str.contains(gun.lower())

    # Filter the DataFrame based on the combined mask
    filtered_df = df[mask]

    return filtered_df

def process_gunshot_data(file_path):
    # Extract the base filename and replace .txt with .wav
    base_name = os.path.basename(file_path)
    wav_filename = os.path.splitext(base_name)[0] + '.wav'

    # Read the file and extract the start times
    start_times = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) >= 1:
                start_time = float(columns[0])
                start_times.append(start_time)

    # Get the length of the list
    list_length = len(start_times)

    # Construct the full path for the wav file
    full_wav_path = os.path.join(os.path.dirname(file_path), wav_filename)

    return full_wav_path, start_times, list_length

def build_dataframe_from_folder(folder_path):
    data = {
        'filename': [],
        'gunshot_location_in_seconds': [],
        'num_gunshots': []
    }

    # Iterate through the folder and find all .txt files
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            wav_filename, start_times, list_length = process_gunshot_data(file_path)

            # Append data to the dataframe columns
            data['filename'].append(wav_filename)
            data['gunshot_location_in_seconds'].append(start_times)
            data['num_gunshots'].append(list_length)

    # Create the DataFrame
    df = pd.DataFrame(data, columns=['filename', 'gunshot_location_in_seconds', 'num_gunshots'])
    return df