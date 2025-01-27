import torch as th
import numpy as np
import importlib
from PopMusicInformationRetrieval import gunshot_utils as utils
from torch.utils.data import Dataset
import torchaudio

importlib.reload(utils)

class GunshotDataset(Dataset):
    def __init__(self, music_df, gunshot_df, music_metadata, gunshot_prob=0.5, num_samples=1000, real_music_gunshot=False, shuffle=True):
        super().__init__()
        self.music_paths = music_df['filename'].tolist()
        self.music_onsets = music_df['onsets'].tolist()
        self.gunshot_paths = gunshot_df['filename'].tolist()
        self.gunshot_truth = gunshot_df['gunshot_location_in_seconds'].tolist()
        self.music_metadata = music_metadata
        self.gunshot_prob = gunshot_prob
        self.num_samples = num_samples
        self.real_music_gunshot = real_music_gunshot
        self.shuffle = shuffle

    def __getitem__(self, idx):
        if not self.shuffle:
            # Use idx directly when shuffle=False
            music_idx = idx % len(self.music_paths)
            gunshot_idx = idx % len(self.gunshot_paths)
        else:
            # Randomly pick indices when shuffle=True
            music_idx = np.random.randint(0, len(self.music_paths))
            gunshot_idx = np.random.randint(0, len(self.gunshot_paths))

        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]

        add_gunshot = (np.random.rand() < self.gunshot_prob) if self.shuffle else (idx % 2 == 0)

        if add_gunshot:
            spectrograms, labels, waveform_segment = self._add_gunshot(fn_music, fn_music_metadata, onset_times, gunshot_idx, idx)
        else:
            spectrograms, labels, waveform_segment = self._add_music_only(fn_music, fn_music_metadata, onset_times)

        if not spectrograms or not labels:
            raise ValueError("Spectrograms or labels are empty after preprocessing")

        return spectrograms[0], labels[0], waveform_segment

    def _get_spectrograms_and_labels(self):
        """Helper function to retrieve spectrograms and labels."""
        add_gunshot = (np.random.rand() < self.gunshot_prob)
        music_idx = np.random.randint(0, len(self.music_paths))
        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]

        if add_gunshot:
            spectrograms, labels = self._add_gunshot(fn_music, fn_music_metadata, onset_times)
        else:
            spectrograms, labels = self._add_music_only(fn_music, fn_music_metadata, onset_times)
        return spectrograms, labels

    def _add_gunshot(self, fn_music, fn_music_metadata, onset_times, gunshot_idx, idx):
        """Handles adding gunshot to music or using a standalone gunshot segment."""
        spectrograms, labels = [], []
        fn_gunshot = self.gunshot_paths[gunshot_idx]
        gunshot_times = self.gunshot_truth[gunshot_idx]

        if self.real_music_gunshot:
            if gunshot_times:
                spectrograms, labels, gunshot_segment = self._process_gunshot_only(fn_gunshot, gunshot_times, idx)
        else:
            spectrograms, labels, gunshot_segment = self._combine_music_and_gunshot(fn_music, fn_music_metadata, fn_gunshot, gunshot_times, onset_times, idx)
        return spectrograms, labels, gunshot_segment

    def _process_gunshot_only(self, fn_gunshot, gunshot_times, idx):
        """Processes a standalone gunshot segment."""
        if not self.shuffle:
            gunshot_time = gunshot_times[idx % len(gunshot_times)]
        else:
            gunshot_time = gunshot_times[np.random.randint(0, len(gunshot_times))]

        gunshot_waveform, sr_gunshot = torchaudio.load(fn_gunshot)
        gunshot_waveform = self.process_gunshot(gunshot_waveform)
        gunshot_segment = utils.select_gunshot_segment(gunshot_waveform, sr_gunshot, gunshot_time)
        spectrograms, labels = utils.preprocess_audio_train(gunshot_segment, label=1)
        return spectrograms, labels, gunshot_segment


    def _combine_music_and_gunshot(self, fn_music, fn_music_metadata, fn_gunshot, gunshot_times, onset_times, idx):
        """Combines music and gunshot into one segment and processes it."""
        spectrograms, labels = [], []
        music_waveform = utils.select_valid_onset_segment(file_path=fn_music, metadata=fn_music_metadata, onset_times=onset_times)

        if gunshot_times:
            if not self.shuffle:
                gunshot_time = gunshot_times[idx % len(gunshot_times)]
            else:
                gunshot_time = gunshot_times[np.random.randint(0, len(gunshot_times))]

            segment, sr = self.combine_music_and_gunshot(music_waveform, fn_gunshot, gunshot_time)
            spectrograms, labels = utils.preprocess_audio_train(segment, label=1)

        return spectrograms, labels, segment


    def _add_music_only(self, fn_music, fn_music_metadata, onset_times):
        """Processes a music segment without a gunshot."""
        music_waveform = utils.select_valid_onset_segment(file_path=fn_music, metadata=fn_music_metadata, onset_times=onset_times)
        spectrograms, labels = utils.preprocess_audio_train(music_waveform, label=0)
        return spectrograms, labels, music_waveform

    def get_random_music_with_gunshot(self, idx=None):
        """
        Returns a random waveform that contains both music and gunshot.
        """
        spectrograms = []
        segment = None
        if self.real_music_gunshot:
            gunshot_idx = np.random.randint(0, len(self.gunshot_paths)) if self.shuffle else idx % len(self.gunshot_paths)
            fn_gunshot = self.gunshot_paths[gunshot_idx]
            gunshot_times = self.gunshot_truth[gunshot_idx]

            if gunshot_times:
                spectrograms, _, segment = self._process_gunshot_only(fn_gunshot, gunshot_times, idx)
        else:
            music_idx = np.random.randint(0, len(self.music_paths)) if self.shuffle else idx % len(self.music_paths)
            fn_music = self.music_paths[music_idx]
            onset_times = self.music_onsets[music_idx]
            fn_music_metadata = self.music_metadata[fn_music]
            music_waveform = utils.select_valid_onset_segment(file_path=fn_music, metadata=fn_music_metadata, onset_times=onset_times)

            gunshot_idx = np.random.randint(0, len(self.gunshot_paths)) if self.shuffle else idx % len(self.gunshot_paths)
            fn_gunshot = self.gunshot_paths[gunshot_idx]
            gunshot_times = self.gunshot_truth[gunshot_idx]

            # Pick a single time from gunshot_times
            if gunshot_times:
                gunshot_time = gunshot_times[idx % len(gunshot_times)] if not self.shuffle else gunshot_times[np.random.randint(0, len(gunshot_times))]
                segment, _ = self.combine_music_and_gunshot(music_waveform, fn_gunshot, gunshot_time)
                spectrograms, _ = utils.preprocess_audio_train(segment, label=1)

        return segment, spectrograms[0] if spectrograms else None

    def get_random_music_onset(self):
        """Returns a random music onset segment."""
        music_idx = np.random.randint(0, len(self.music_paths))
        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]
        music_waveform = utils.select_valid_onset_segment(file_path=fn_music, metadata=fn_music_metadata, onset_times=onset_times)
        music_spectrograms, _ = utils.preprocess_audio_train(music_waveform, label=0)
        return music_waveform, music_spectrograms[0] if music_spectrograms else None

    def loudness_in_db(self, waveform):
        """
        Calculate the loudness of the audio waveform in decibels (dB).
        """
        rms = th.sqrt(th.mean(waveform**2))
        db = 20 * th.log10(rms + 1e-6)  # Adding small constant to avoid log(0)
        return db.item()

    def normalize_to_target_loudness(self, waveform, target_db=-15):
        """
        Normalize the audio waveform to the target loudness in dB.
        """
        current_db = self.loudness_in_db(waveform)
        gain_db = target_db - current_db
        gain_factor = 10 ** (gain_db / 20)  # Convert dB to linear scale
        return waveform * gain_factor

    def compress_dynamic_range(self, waveform, threshold_db=-10, ratio=4.0):
        """
        Apply dynamic range compression to the audio waveform.
        """
        threshold = 10 ** (threshold_db / 20)  # Convert dB threshold to linear scale
        waveform_abs = th.abs(waveform)
        compressed = th.where(
            waveform_abs > threshold,
            threshold + (waveform_abs - threshold) / ratio,
            waveform_abs,
            )
        return th.sign(waveform) * compressed

    def process_gunshot(self, waveform, target_range=(-20, -10)):
        """
        Bring the gunshot to the target decibel range.
        - Normalize to the lower bound of the range.
        - Apply compression for overblown waveforms.
        - Scale the waveform randomly within the range.
        """
        # Step 1: Normalize to lower bound
        target_db = target_range[0]
        normalized_waveform = self.normalize_to_target_loudness(waveform, target_db)

        # Step 2: Compress dynamic range for overblown sounds
        compressed_waveform = self.compress_dynamic_range(normalized_waveform, threshold_db=target_db)

        random_db = np.random.uniform(*target_range)  # Random dB in the range
        processed_waveform = self.normalize_to_target_loudness(compressed_waveform, random_db)

        return processed_waveform

    def combine_music_and_gunshot(self, music_waveform, gunshot_file, gunshot_time, sample_rate=44100, pre_gunshot_time=0):
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
        gunshot_waveform = self.process_gunshot(gunshot_waveform)

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

        # --- DEALING WITH THE MUSIC AND GUNSHOT OVERLAY ---
        # Overlay the gunshot onto the music from the beginning
        combined_segment = music_waveform.clone()
        combined_segment[:, :gunshot_segment.size(1)] += gunshot_segment

        return combined_segment, sample_rate

    def __len__(self):
        return self.num_samples
