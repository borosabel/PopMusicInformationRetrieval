import random
import numpy as np
import torchaudio
import torch as th
from tqdm import tqdm
import librosa
from pydub import AudioSegment
import re
import ast
import sounddevice as sd
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

def plot_waveform(waveform, sample_rate, vertical_lines=None):
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
        plt.show()
    except Exception as e:
        print(f"Error occurred while plotting waveform: {e}")


def play_audio(waveform, sample_rate):
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


def combine_music_and_gunshot(music_file, gunshot_file, gunshot_time, excerpt_len_sec=5.0, gunshot_placement_sec=2.0,
                              gunshot_volume_increase_dB=5, sample_rate=44100, pre_gunshot_time=0.5):
    """
    Combines a music segment with a gunshot at the specified placement time.
    """
    # --- DEALING WITH THE MUSIC FILE ---
    music_waveform, sr = torchaudio.load(music_file)
    if sr != sample_rate:
        # print(f"Resampling music from {sr} Hz to {sample_rate} Hz. \n")
        music_waveform = torchaudio.transforms.Resample(sr, sample_rate)(music_waveform)

    excerpt_len_samples = int(excerpt_len_sec * sample_rate)

    # Ensure the music segment is within bounds
    total_music_samples = music_waveform.size(1)
    max_start_sample = max(0, total_music_samples - excerpt_len_samples)
    start_pos_music = random.randint(0, max_start_sample)
    music_segment = music_waveform[:, start_pos_music:start_pos_music + excerpt_len_samples]
    # --- DEALING WITH THE MUSIC FILE ---

    # --- DEALING WITH THE GUNSHOT FILE ---
    # print("Loading the gunshot file...\n")
    gunshot_waveform, sr_gunshot = torchaudio.load(gunshot_file)
    if sr_gunshot != sample_rate:
        # print(f"Resampling gunshot from {sr_gunshot} Hz to {sample_rate} Hz.\n")
        gunshot_waveform = torchaudio.transforms.Resample(sr_gunshot, sample_rate)(gunshot_waveform)

    if gunshot_time >= pre_gunshot_time:
        gunshot_start_sample = int((gunshot_time - pre_gunshot_time) * sample_rate)
        gunshot_segment = gunshot_waveform[:, gunshot_start_sample:]
    else:
        # If gunshot_time is less than the threshold, keep the entire waveform
        gunshot_segment = gunshot_waveform

    # Apply volume gain to gunshot
    gain_factor = 10 ** (gunshot_volume_increase_dB / 20)
    gunshot_segment *= gain_factor

    # print(f"Applying a {gunshot_volume_increase_dB} dB volume increase to the gunshot.")

    gunshot_placement_sample = int(gunshot_placement_sec * sample_rate)

    if gunshot_placement_sample + gunshot_segment.size(1) > music_segment.size(1):
        gunshot_segment = gunshot_segment[:, :music_segment.size(1) - gunshot_placement_sample]
    # --- DEALING WITH THE GUNSHOT FILE ---

    # Overlay the gunshot onto the music
    combined_segment = music_segment.clone()
    combined_segment[:, gunshot_placement_sample:gunshot_placement_sample + gunshot_segment.size(1)] += gunshot_segment

    return combined_segment, sample_rate

def preprocess_audio_train(waveform, sample_rate, label, gunshot_time=None):
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
    # print("Waveform shape: ", waveform.shape)
    # print("Sampling rate: ", sample_rate)
    # If it's a gunshot sample, use the select_gunshot_segment function
    if label == 1 and gunshot_time is not None:
        segment = select_gunshot_segment(waveform, sample_rate, gunshot_time, FRAME_LENGTH)
    else:
        segment = select_random_segment(waveform, sample_rate, FRAME_LENGTH)
    # print(f"Segment shape after cutting {FRAME_LENGTH} size: {segment.shape}")
    mel_specgram = calculate_melbands(segment[0], sample_rate)
    # print(f"MEL SPECTOGRAM shape of the segment {mel_specgram.shape}")
    spectrograms.append(mel_specgram)
    labels.append(label)
    # print("------PREPROCESSING AUDIO DATA------")
    return spectrograms, labels



def select_random_segment(waveform, sample_rate, frame_length):
    total_duration = waveform.size(1) / sample_rate
    segment_length = frame_length

    if total_duration * sample_rate <= frame_length:
        return waveform

    start_time = random.uniform(0, total_duration - (frame_length / sample_rate))
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + segment_length

    return waveform[:, start_sample:end_sample]


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
    print(f"Frame count: {frame_count} Number of frames: {frame_count} corresponds to {seconds:.4f} seconds at a sample rate of {sample_rate} Hz.")
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

    mel_specs = th.log10(th.stack(mel_specs) + 1e-08)  # Shape: [3, 80, NUM FRAMES]

    # Calculate additional timbre features
    waveform_np = waveform.numpy()
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform_np, sr=sample_rate)
    spectral_flatness = librosa.feature.spectral_flatness(y=waveform_np)
    # MFCCs can be added similarly if required

    # Convert to tensors and ensure they match the shape along the time axis
    spectral_centroid = th.tensor(spectral_centroid, dtype=th.float32).unsqueeze(0)
    spectral_bandwidth = th.tensor(spectral_bandwidth, dtype=th.float32).unsqueeze(0)
    spectral_flatness = th.tensor(spectral_flatness, dtype=th.float32).unsqueeze(0)

    # Concatenate additional features along the feature axis
    additional_features = th.cat([spectral_centroid, spectral_bandwidth, spectral_flatness], dim=0)  # Shape: [3, 1, 15]

    # Resize additional features to match the mel spectrogram's feature dimension (if necessary)
    additional_features = additional_features.permute(1, 0, 2)

    # Ensure both tensors have the same shape along the feature and time dimensions
    mel_specs = mel_specs.permute(1, 0, 2)  # Shape: [80, 3, 15]
    combined_features = th.cat([mel_specs, additional_features], dim=0)
    combined_features = combined_features.permute(1, 0, 2)

    return combined_features

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
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training")

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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

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