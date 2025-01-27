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

            segment, sr = utils.combine_music_and_gunshot(music_waveform, fn_gunshot, gunshot_time, gunshot_volume_increase_dB=0)
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
            gunshot_waveform, sr = torchaudio.load(fn_gunshot)
            gunshot_times = self.gunshot_truth[gunshot_idx]

            # Pick a single time from gunshot_times
            if gunshot_times:
                gunshot_time = gunshot_times[idx % len(gunshot_times)] if not self.shuffle else gunshot_times[np.random.randint(0, len(gunshot_times))]
                segment, _ = utils.combine_music_and_gunshot(music_waveform, fn_gunshot, gunshot_time)
                spectrograms, _ = utils.preprocess_audio_train(segment, label=1)

        return segment, gunshot_waveform, spectrograms[0] if spectrograms else None

    def get_random_music_onset(self):
        """Returns a random music onset segment."""
        music_idx = np.random.randint(0, len(self.music_paths))
        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]
        music_waveform = utils.select_valid_onset_segment(file_path=fn_music, metadata=fn_music_metadata, onset_times=onset_times)
        music_spectrograms, _ = utils.preprocess_audio_train(music_waveform, label=0)
        return music_waveform, music_spectrograms[0] if music_spectrograms else None

    def __len__(self):
        return self.num_samples