import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal


class EEGDataLoader:
    """Handles loading and preprocessing of EEG data for feature extraction"""

    def __init__(self, config):
        """Initialize with configuration"""
        print("\nInitializing EEGDataLoader...")
        self.labels = config.get_labels()
        self.required_columns = config.get_channels()
        self.sfreq = 250  # Target sampling frequency
        self.epoch_length = 1.0  # 1-second epochs for feature extraction
        self.X = []
        self.y = []

        # Validate required columns
        if (
            not isinstance(self.required_columns, list)
            or len(self.required_columns) != 31
        ):
            raise ValueError("Must provide exactly 31 channel names")
        print(
            f"Processing {len(self.labels)} files with {len(self.required_columns)} channels"
        )

    def _load_single_file(self, file_path):
        """Load and validate a single EEG file"""
        try:
            print(f"Loading {Path(file_path).name}...")
            df = pd.read_csv(file_path)

            # Validate columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            return df[self.required_columns].values.T  # (channels, time)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def _preprocess_eeg(self, raw_data, original_sfreq=500):
        """Apply standard EEG preprocessing pipeline"""
        print("Applying preprocessing...")

        # Create MNE Raw object
        info = mne.create_info(
            ch_names=self.required_columns, sfreq=original_sfreq, ch_types="eeg"
        )
        raw = mne.io.RawArray(raw_data, info)

        # Preprocessing steps
        raw.filter(1, 40, fir_design="firwin")  # Bandpass filter
        raw.notch_filter(50)  # Notch filter for line noise

        # Resample if needed
        if original_sfreq != self.sfreq:
            raw.resample(self.sfreq)

        return raw.get_data()  # (n_chans, n_times)

    def _create_epochs(self, data):
        """Segment continuous EEG into fixed-length epochs"""
        print("Segmenting into epochs...")
        epoch_samples = int(self.epoch_length * self.sfreq)
        n_epochs = data.shape[1] // epoch_samples

        epochs = []
        for i in range(n_epochs):
            start = i * epoch_samples
            end = start + epoch_samples
            epochs.append(data[:, start:end])

        return np.array(epochs)  # (n_epochs, n_chans, n_samples)

    def _remove_bad_epochs(self, epochs, threshold=100):
        """Remove epochs with excessive amplitude"""
        print("Removing bad epochs...")
        good_epochs = []
        for epoch in epochs:
            if np.max(np.abs(epoch)) < threshold:
                good_epochs.append(epoch)
        return np.array(good_epochs)

    def load_all_data(self):
        """Main method to load and process all files"""
        print("\n=== Starting Data Loading ===")
        for file_path, label in self.labels.items():
            raw_data = self._load_single_file(file_path)
            if raw_data is None:
                continue

            # Process and segment
            clean_data = self._preprocess_eeg(raw_data)
            epochs = self._create_epochs(clean_data)
            clean_epochs = self._remove_bad_epochs(epochs)

            self.X.extend(clean_epochs)
            self.y.extend([label] * len(clean_epochs))

        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        print(f"\nFinal dataset shape: {self.X.shape}")
        print("Class distribution:")
        print(f"  Class 0 (No stress): {np.sum(self.y == 0)} samples")
        print(f"  Class 1 (Stress): {np.sum(self.y == 1)} samples")

        return self.X, self.y

    def get_channel_names(self):
        """Return the list of channel names"""
        return self.required_columns
