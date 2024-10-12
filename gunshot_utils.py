import random
import numpy as np
import pandas as pd
import torchaudio
import torch as th
from tqdm import tqdm
import librosa
import os
from pydub import AudioSegment
import re
import ast
from pydub.playback import play
from IPython.display import Audio

SAMPLING_RATE = 44100
HOP_LENGTH = 512  # Define the hop length (e.g., 512 samples)
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


def play_audio(waveform, sample_rate):
    return Audio(waveform.numpy(), rate=sample_rate)


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

import torchaudio
import random
import IPython.display as ipd

def combine_music_and_gunshot(music_file, gunshot_file, gunshot_time, excerpt_len_sec=5.0, gunshot_placement_sec=2.0,
                              gunshot_volume_increase_dB=5, sample_rate=44100, pre_gunshot_time=0.5):
    """
    Combines a music segment with a gunshot at the specified placement time.
    """
    music_waveform, sr = torchaudio.load(music_file)
    if sr != sample_rate:
        print(f"Resampling music from {sr} Hz to {sample_rate} Hz.")
        music_waveform = torchaudio.transforms.Resample(sr, sample_rate)(music_waveform)

    # Play the original music file (convert tensor to 1D for playback)
    music_audio_preview = music_waveform.mean(dim=0).numpy() if music_waveform.ndim > 1 else music_waveform.numpy()
    # print("Here is a preview of the original music:")
    # ipd.display(ipd.Audio(music_audio_preview, rate=sample_rate))

    excerpt_len_samples = int(excerpt_len_sec * sample_rate)

    # Ensure the music segment is within bounds
    total_music_samples = music_waveform.size(1)
    max_start_sample = max(0, total_music_samples - excerpt_len_samples)
    start_pos_music = random.randint(0, max_start_sample)
    music_segment = music_waveform[:, start_pos_music:start_pos_music + excerpt_len_samples]

    print(f"Extracted a {excerpt_len_sec} seconds music excerpt for processing.")

    # Play the music segment
    music_segment_preview = music_segment.mean(dim=0).numpy() if music_segment.ndim > 1 else music_segment.numpy()
    # print("Preview of the music segment that will be combined with the gunshot:")
    # ipd.display(ipd.Audio(music_segment_preview, rate=sample_rate))

    print("Loading the gunshot file...")
    gunshot_waveform, sr_gunshot = torchaudio.load(gunshot_file)
    if sr_gunshot != sample_rate:
        print(f"Resampling gunshot from {sr_gunshot} Hz to {sample_rate} Hz.")
        gunshot_waveform = torchaudio.transforms.Resample(sr_gunshot, sample_rate)(gunshot_waveform)

    gunshot_start_sample = int((gunshot_time - pre_gunshot_time) * sample_rate)
    gunshot_segment = gunshot_waveform[:, gunshot_start_sample:]

    # Apply volume gain to gunshot
    gain_factor = 10 ** (gunshot_volume_increase_dB / 20)
    gunshot_segment *= gain_factor

    print(f"Applying a {gunshot_volume_increase_dB} dB volume increase to the gunshot.")

    # Play the gunshot segment
    gunshot_segment_preview = gunshot_segment.mean(dim=0).numpy() if gunshot_segment.ndim > 1 else gunshot_segment.numpy()
    # print("Preview of the gunshot that will be added to the music:")
    # ipd.display(ipd.Audio(gunshot_segment_preview, rate=sample_rate))

    gunshot_placement_sample = int(gunshot_placement_sec * sample_rate)

    if gunshot_placement_sample + gunshot_segment.size(1) > music_segment.size(1):
        gunshot_segment = gunshot_segment[:, :music_segment.size(1) - gunshot_placement_sample]

    # Overlay the gunshot onto the music
    combined_segment = music_segment.clone()
    combined_segment[:, gunshot_placement_sample:gunshot_placement_sample + gunshot_segment.size(1)] += gunshot_segment

    print(f"Combining the music and gunshot. The gunshot will be placed at {gunshot_placement_sec} seconds.")

    # Play the final combined audio
    combined_segment_preview = combined_segment.mean(dim=0).numpy() if combined_segment.ndim > 1 else combined_segment.numpy()
    # print("Here is the final combined audio with the music and the gunshot:")
    # ipd.display(ipd.Audio(combined_segment_preview, rate=sample_rate))

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
    print("------PREPROCESSING AUDIO DATA------")
    spectrograms = []
    labels = []
    print("Waveform shape: ", waveform.shape)
    print("Sampling rate: ", sample_rate)
    # If it's a gunshot sample, use the select_gunshot_segment function
    if label == 1 and gunshot_time is not None:
        segment = select_gunshot_segment(waveform, sample_rate, gunshot_time, FRAME_LENGTH)
    else:
        segment = select_random_segment(waveform, sample_rate, FRAME_LENGTH)
    print(f"Segment shape after cutting {FRAME_LENGTH} size: {segment.shape}")
    mel_specgram = calculate_melbands(segment[0], sample_rate)
    print(f"MEL SPECTOGRAM shape of the segment {mel_specgram.shape}")
    spectrograms.append(mel_specgram)
    labels.append(label)
    print("------PREPROCESSING AUDIO DATA------")
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


def select_gunshot_segment(waveform, sample_rate, gunshot_time, frame_length, max_shift_sec=0):
    """
    Selects a segment of audio around the gunshot, allowing for a random shift in the gunshot position.

    Parameters:
        waveform (Tensor): The audio waveform tensor.
        sample_rate (int): The sample rate of the audio.
        gunshot_time (float): The time of the gunshot in seconds.
        frame_length (int): The desired length of the segment (in samples).
        max_shift_sec (float): Maximum time in seconds by which to shift the gunshot position.

    Returns:
        Tensor: The selected segment of audio.
    """
    print(f"------SELECTING GUNSHOT SEGMENT WITH FRAME SIZE OF {frame_length}------")
    # Compute the amount of shift in time (in seconds) randomly within [-max_shift_sec, +max_shift_sec]
    # random_shift = random.uniform(-max_shift_sec, max_shift_sec)

    # Adjust the gunshot start time with the random shift
    # shifted_gunshot_time = gunshot_time + random_shift

    # Ensure the start time doesn't go out of bounds (start time should be >= 0)
    start_time = max(0, gunshot_time - (frame_length / sample_rate) / 2)

    # Ensure the end time doesn't exceed the waveform length
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + int(frame_length)

    # Ensure we don't exceed the total length of the waveform
    end_sample = min(end_sample, waveform.size(1))
    start_sample = max(0, end_sample - int(frame_length))  # Adjust start if needed
    print(f"------SELECTING GUNSHOT SEGMENT------")
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

    mel_specs = th.log10(th.stack(mel_specs) + 1e-08)  # Shape: [3, 80, 15]

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


##########################################################################################################################

def preprocess_audio(files):
    spectrograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_specgram = calculate_melbands(waveform[0], sample_rate)
        spectrograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectrograms, sample_rates

#%%
