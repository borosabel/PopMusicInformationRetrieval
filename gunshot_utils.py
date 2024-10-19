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
import minimp3py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

SAMPLING_RATE = 44100
HOP_LENGTH = 512
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050
WIN_LENGTHS = [1024, 2048, 4096]
WIN_SIZES = [0.023, 0.046, 0.093]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
NUM_FRAMES = 50
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


def resample_and_convert_to_wav(source_folder, target_folder, target_sample_rate=SAMPLING_RATE):
    """
    Resample all audio files to the target sample rate and save them as WAV in the target folder.

    :param source_folder: Folder containing the original audio files.
    :param target_folder: Folder where the resampled WAV files will be saved.
    :param target_sample_rate: The sample rate to which all audio should be resampled.
    """

    # Ensure the target directory exists
    os.makedirs(target_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".mp3") or file_name.endswith(".wav"):
            source_file_path = os.path.join(source_folder, file_name)
            target_file_path = os.path.join(target_folder, os.path.splitext(file_name)[0] + '.wav')

            try:
                # Load the audio file
                audio_waveform, original_sample_rate = torchaudio.load(source_file_path)

                # Resample if necessary
                if original_sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate,
                                                               new_freq=target_sample_rate)
                    audio_waveform = resampler(audio_waveform)
                    print(f"Resampling {file_name} from {original_sample_rate} Hz to {target_sample_rate} Hz")
                else:
                    print(f"Sample rate of {file_name} is already {target_sample_rate} Hz. No resampling needed.")

                torchaudio.save(target_file_path, audio_waveform, target_sample_rate)
                print(f"Saved {target_file_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


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


def combine_music_and_gunshot(music_waveform, gunshot_file, gunshot_time, gunshot_volume_increase_dB=5, sample_rate=44100, pre_gunshot_time=0):
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

    # If the gunshot segment is shorter than the music, pad with zeros
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


def train_model(model, optimizer, criterion, train_loader, valid_loader, epochs=10, mean=None, std=None, patience=3):
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided for normalization.")

    mean = mean.to(device)
    std = std.to(device)
    model = model.to(device)
    best_score = 0.0
    epochs_since_improvement = 0

    if use_cuda:
        scaler = th.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Add tqdm progress bar for training loop
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training")

        for features, labels in train_loader_tqdm:
            features, labels = features.to(device), labels.to(device).float().to(device)
            optimizer.zero_grad()
            features = (features - mean) / std

            if use_cuda:
                with th.cuda.amp.autocast():
                    outputs = model(features).view(-1)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features).view(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * features.size(0)

            # Update tqdm description with current loss
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        val_score = evaluate_model_simple(model, valid_loader, mean, std)

        if val_score > best_score:
            best_score = val_score
            epochs_since_improvement = 0
            print(f"New best ROC AUC score: {best_score:.4f}, model saved.")
            # Save the model if desired
            # torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"No improvement in ROC AUC score for {patience} epochs. Stopping training.")
            break

    # After training, find the optimal threshold
    best_threshold = find_optimal_threshold_after_training(model, valid_loader, mean, std)

    # Compute and display the confusion matrix
    cm = compute_confusion_matrix(model, valid_loader, best_threshold, mean, std)
    display_confusion_matrix(cm)

    return best_threshold, best_score


def evaluate_model_simple(model, valid_loader, mean, std):
    all_outputs = []
    all_labels = []

    valid_loader_tqdm = tqdm(valid_loader, desc="Validation")

    with th.no_grad():
        for features, labels in valid_loader_tqdm:
            features = features.to(device)
            labels = labels.to(device).float()
            features = (features - mean) / std
            outputs = model(features).view(-1).cpu().numpy()
            all_outputs.append(outputs)
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_outputs)
    print(f"Validation ROC AUC: {auc:.4f}")
    return auc


def find_optimal_threshold_after_training(model, valid_loader, mean, std):
    all_outputs = []
    all_labels = []

    # Add tqdm progress bar for validation loop
    valid_loader_tqdm = tqdm(valid_loader, desc="Finding Optimal Threshold")

    with th.no_grad():
        for features, labels in valid_loader_tqdm:
            features = features.to(device)
            labels = labels.to(device).float()
            features = (features - mean) / std
            outputs = model(features).view(-1).cpu().numpy()
            all_outputs.append(outputs)
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    youdens_j = tpr - fpr
    idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[idx]

    print(f"Optimal threshold found: {optimal_threshold:.4f}")
    return optimal_threshold


def compute_confusion_matrix(model, valid_loader, threshold, mean, std):
    """
    Compute confusion matrix using batch processing.

    Parameters:
        model (torch.nn.Module): The trained model.
        valid_loader (DataLoader): DataLoader for validation data.
        threshold (float): Threshold to convert probabilities to binary predictions.
        mean (torch.Tensor): Mean for normalization.
        std (torch.Tensor): Standard deviation for normalization.

    Returns:
        cm (numpy.ndarray): Confusion matrix.
    """
    all_outputs = []
    all_labels = []

    # Add tqdm progress bar for validation loop
    valid_loader_tqdm = tqdm(valid_loader, desc="Computing Confusion Matrix")

    with th.no_grad():
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
    """
    Displays the confusion matrix using matplotlib.

    Parameters:
        cm (numpy.ndarray): Confusion matrix.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='magma')
    plt.title('Confusion Matrix')
    plt.show()
