import torch as th
import numpy as np
import importlib
import torchaudio
from torch.utils.data import Dataset

# Adjust this import path if needed
from PopMusicInformationRetrieval import gunshot_utils as utils

importlib.reload(utils)

class GunshotDataset(Dataset):
    def __init__(self, music_df, gunshot_df, music_metadata,
                 gunshot_prob=0.5, num_samples=1000,
                 real_music_gunshot=False, shuffle=True):
        """
        - music_df, gunshot_df: DataFrames with 'filename' columns.
        - music_metadata: dict keyed by filename with 'sample_rate', 'num_frames', etc.
        - gunshot_prob: Probability of adding a gunshot.
        - num_samples: Length of dataset.
        - real_music_gunshot: If True, uses actual gunshot segments from real recordings.
        - shuffle: If True, picks random indices each time.
        """
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
        # 1. Pick random or deterministic indices
        if self.shuffle:
            music_idx = np.random.randint(0, len(self.music_paths))
            gunshot_idx = np.random.randint(0, len(self.gunshot_paths))
        else:
            music_idx = idx % len(self.music_paths)
            gunshot_idx = idx % len(self.gunshot_paths)

        # 2. Load music info
        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]

        # 3. Decide whether to add gunshot
        add_gunshot = (np.random.rand() < self.gunshot_prob) if self.shuffle else (idx % 2 == 0)

        # 4. Process either with or without gunshot
        if add_gunshot:
            spectrograms, labels, waveform_segment = self._add_gunshot(
                fn_music, fn_music_metadata, onset_times, gunshot_idx, idx
            )
        else:
            spectrograms, labels, waveform_segment = self._add_music_only(
                fn_music, fn_music_metadata, onset_times
            )

        # Sanity check
        if not spectrograms or not labels:
            raise ValueError("Spectrograms or labels are empty after preprocessing")

        # Return the first spectrogram + label + raw segment
        return spectrograms[0], labels[0], waveform_segment

    def __len__(self):
        return self.num_samples

    # ---------------------------------------------------
    #                 INTERNAL METHODS
    # ---------------------------------------------------
    def _add_gunshot(self, fn_music, fn_music_metadata, onset_times, gunshot_idx, idx):
        """
        Either use a 'real music + gunshot' or artificially combine them.
        """
        spectrograms, labels = [], []
        fn_gunshot = self.gunshot_paths[gunshot_idx]
        gunshot_times = self.gunshot_truth[gunshot_idx]

        if self.real_music_gunshot:
            if gunshot_times:
                # actual gunshot segment only
                spectrograms, labels, gunshot_segment = self._process_gunshot_only(
                    fn_gunshot, gunshot_times, idx
                )
        else:
            # artificially combine music + gunshot
            spectrograms, labels, gunshot_segment = self._combine_music_and_gunshot(
                fn_music, fn_music_metadata, fn_gunshot, gunshot_times, onset_times, idx
            )
        return spectrograms, labels, gunshot_segment

    def _process_gunshot_only(self, fn_gunshot, gunshot_times, idx):
        """
        Loads a single gunshot waveform and selects the relevant segment.
        """
        # pick gunshot time
        if self.shuffle:
            gunshot_time = np.random.choice(gunshot_times)
        else:
            gunshot_time = gunshot_times[idx % len(gunshot_times)]

        gunshot_waveform, sr_gunshot = torchaudio.load(fn_gunshot)
        gunshot_waveform = self.process_gunshot(gunshot_waveform)
        gunshot_segment = utils.select_gunshot_segment(
            gunshot_waveform, sr_gunshot, gunshot_time
        )
        spectrograms, labels = utils.preprocess_audio_train(gunshot_segment, label=1)
        return spectrograms, labels, gunshot_segment

    def _combine_music_and_gunshot(self, fn_music, fn_music_metadata, fn_gunshot,
                                   gunshot_times, onset_times, idx):
        """
        Combine a valid music onset segment with a gunshot at a chosen time.
        """
        spectrograms, labels = [], []
        # First, get a valid music segment
        music_waveform = utils.select_valid_onset_segment(
            file_path=fn_music,
            metadata=fn_music_metadata,
            onset_times=onset_times
        )
        if gunshot_times:
            if self.shuffle:
                gunshot_time = np.random.choice(gunshot_times)
            else:
                gunshot_time = gunshot_times[idx % len(gunshot_times)]

            # Combine them
            segment, sr = self.combine_music_and_gunshot(
                music_waveform, fn_gunshot, gunshot_time
            )
            spectrograms, labels = utils.preprocess_audio_train(segment, label=1)

        return spectrograms, labels, segment

    def _add_music_only(self, fn_music, fn_music_metadata, onset_times):
        """
        Just returns a music segment with no gunshots.
        """
        music_waveform = utils.select_valid_onset_segment(
            file_path=fn_music,
            metadata=fn_music_metadata,
            onset_times=onset_times
        )
        spectrograms, labels = utils.preprocess_audio_train(music_waveform, label=0)
        return spectrograms, labels, music_waveform

    # ---------------------------------------------------
    #             PUBLIC HELPER METHODS
    # ---------------------------------------------------
    def get_random_music_with_gunshot(self, idx=None):
        """
        Returns a random overlay of music and gunshot.
        """
        spectrograms = []
        segment = None

        if self.real_music_gunshot:
            # Actual gunshot-only wave
            gunshot_idx = (np.random.randint(0, len(self.gunshot_paths))
                           if self.shuffle else idx % len(self.gunshot_paths))
            fn_gunshot = self.gunshot_paths[gunshot_idx]
            gunshot_times = self.gunshot_truth[gunshot_idx]
            if gunshot_times:
                spectrograms, _, segment = self._process_gunshot_only(fn_gunshot, gunshot_times, idx)
        else:
            # Artificial overlay
            music_idx = (np.random.randint(0, len(self.music_paths))
                         if self.shuffle else idx % len(self.music_paths))
            fn_music = self.music_paths[music_idx]
            onset_times = self.music_onsets[music_idx]
            fn_music_metadata = self.music_metadata[fn_music]
            music_waveform = utils.select_valid_onset_segment(fn_music, fn_music_metadata, onset_times)

            gunshot_idx = (np.random.randint(0, len(self.gunshot_paths))
                           if self.shuffle else idx % len(self.gunshot_paths))
            fn_gunshot = self.gunshot_paths[gunshot_idx]
            gunshot_times = self.gunshot_truth[gunshot_idx]

            if gunshot_times:
                gunshot_time = (gunshot_times[idx % len(gunshot_times)]
                                if not self.shuffle else np.random.choice(gunshot_times))
                segment, _ = self.combine_music_and_gunshot(music_waveform, fn_gunshot, gunshot_time)
                spectrograms, _ = utils.preprocess_audio_train(segment, label=1)

        # Return the final overlay
        return segment, spectrograms[0] if spectrograms else None

    def get_random_music_onset(self):
        """
        Returns a random music segment (no gunshot) from a valid onset.
        """
        music_idx = np.random.randint(0, len(self.music_paths))
        fn_music = self.music_paths[music_idx]
        onset_times = self.music_onsets[music_idx]
        fn_music_metadata = self.music_metadata[fn_music]

        music_waveform = utils.select_valid_onset_segment(
            file_path=fn_music,
            metadata=fn_music_metadata,
            onset_times=onset_times
        )
        music_spectrograms, _ = utils.preprocess_audio_train(music_waveform, label=0)
        return music_waveform, music_spectrograms[0] if music_spectrograms else None

    # ---------------------------------------------------
    #           AUDIO PROCESSING PIPELINE
    # ---------------------------------------------------
    def loudness_in_db(self, waveform):
        rms = th.sqrt(th.mean(waveform ** 2))
        db = 20 * th.log10(rms + 1e-6)
        return db.item()

    def normalize_to_target_loudness(self, waveform, target_db=-15):
        current_db = self.loudness_in_db(waveform)
        gain_db = target_db - current_db
        gain_factor = 10 ** (gain_db / 20)
        return waveform * gain_factor

    def compress_dynamic_range(self, waveform, threshold_db=-10, ratio=4.0):
        threshold = 10 ** (threshold_db / 20)
        waveform_abs = th.abs(waveform)
        compressed = th.where(
            waveform_abs > threshold,
            threshold + (waveform_abs - threshold) / ratio,
            waveform_abs
        )
        return th.sign(waveform) * compressed

    def process_gunshot(self, waveform, target_range=(-20, -10)):
        # 1) Normalize to the lower bound
        target_db = target_range[0]
        normalized = self.normalize_to_target_loudness(waveform, target_db)

        # 2) Compress
        compressed = self.compress_dynamic_range(normalized, threshold_db=target_db)

        # 3) Random final loudness in the range
        random_db = np.random.uniform(*target_range)
        processed = self.normalize_to_target_loudness(compressed, random_db)
        return processed

    def combine_music_and_gunshot(self, music_waveform, gunshot_file,
                                  gunshot_time, sample_rate=44100, pre_gunshot_time=0):
        """
        Overlay a processed gunshot onto music_waveform from the start.
        """
        # Load gunshot
        gunshot_waveform, _ = torchaudio.load(gunshot_file)
        gunshot_waveform = self.process_gunshot(gunshot_waveform)

        # Slice the gunshot to match the music length
        if gunshot_time >= pre_gunshot_time:
            gunshot_start_sample = int((gunshot_time - pre_gunshot_time) * sample_rate)
        else:
            gunshot_start_sample = 0

        music_length_samples = music_waveform.size(1)
        gunshot_segment = gunshot_waveform[
                          :, gunshot_start_sample:gunshot_start_sample + music_length_samples
                          ]

        # Pad if shorter
        if gunshot_segment.size(1) < music_length_samples:
            pad_length = music_length_samples - gunshot_segment.size(1)
            gunshot_segment = th.nn.functional.pad(gunshot_segment, (0, pad_length))

        # Overlay
        combined_segment = music_waveform.clone()
        combined_segment[:, :gunshot_segment.size(1)] += gunshot_segment
        return combined_segment, sample_rate
