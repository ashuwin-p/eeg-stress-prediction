import numpy as np
import pywt
from scipy import stats
from scipy.signal import welch

class FeatureExtractor:
    """Handles different feature extraction methods for EEG data"""

    @staticmethod
    def extract_time_domain_features(X):
        """Extract time domain features"""
        n_samples = X.shape[0]
        features = np.zeros((n_samples, 8 * X.shape[1]))  # 8 features × n_channels

        for sample_idx in range(n_samples):
            for channel_idx in range(X.shape[1]):
                epoch = X[sample_idx, channel_idx, :]
                features[sample_idx, channel_idx * 8 : (channel_idx + 1) * 8] = [
                    np.mean(epoch),
                    np.std(epoch),
                    stats.skew(epoch),
                    stats.kurtosis(epoch),
                    np.median(epoch),
                    np.max(epoch) - np.min(epoch),  # Range
                    np.percentile(epoch, 25),  # Q1
                    np.percentile(epoch, 75),  # Q3
                ]
        return features

    @staticmethod
    def extract_frequency_domain_features(X, sfreq=250):
        """Extract frequency domain features"""
        n_samples = X.shape[0]
        n_bands = 5
        n_features = (n_bands + 3) * X.shape[1]  # 5 bands + SEF + mean + peak
        features = np.zeros((n_samples, n_features))

        for sample_idx in range(n_samples):
            for channel_idx in range(X.shape[1]):
                epoch = X[sample_idx, channel_idx, :]
                freqs, psd = welch(epoch, fs=sfreq, nperseg=min(64, len(epoch)))

                # Band power features
                bands = {
                    "delta": (0.5, 4),
                    "theta": (4, 8),
                    "alpha": (8, 13),
                    "beta": (13, 30),
                    "gamma": (30, 45),
                }

                band_powers = []
                for band, (fmin, fmax) in bands.items():
                    band_mask = (freqs >= fmin) & (freqs <= fmax)
                    band_powers.append(np.sum(psd[band_mask]))

                # Spectral edge frequency
                total_power = np.sum(psd)
                sef = 0
                cum_power = 0
                for i, p in enumerate(psd):
                    cum_power += p
                    if cum_power >= 0.95 * total_power:
                        sef = freqs[i]
                        break

                # Store all features for this channel
                start_idx = channel_idx * (n_bands + 3)
                features[sample_idx, start_idx : start_idx + n_bands] = band_powers
                features[sample_idx, start_idx + n_bands] = sef
                features[sample_idx, start_idx + n_bands + 1] = np.mean(psd)
                features[sample_idx, start_idx + n_bands + 2] = freqs[np.argmax(psd)]

        return features

    @staticmethod
    def extract_wavelet_features(X, wavelet="db4", level=4):
        """Extract wavelet transform features"""
        n_samples = X.shape[0]
        n_stats = 6  # mean, std, median, range, skew, kurtosis
        n_features = n_stats * (level + 1) * X.shape[1]
        features = np.zeros((n_samples, n_features))

        for sample_idx in range(n_samples):
            for channel_idx in range(X.shape[1]):
                epoch = X[sample_idx, channel_idx, :]
                coeffs = pywt.wavedec(epoch, wavelet, level=level)

                # Flatten all coefficients with stats
                coeff_features = []
                for coeff in coeffs:
                    coeff_features.extend(
                        [
                            np.mean(coeff),
                            np.std(coeff),
                            np.median(coeff),
                            np.max(coeff) - np.min(coeff),
                            stats.skew(coeff),
                            stats.kurtosis(coeff),
                        ]
                    )

                # Store in correct position
                start_idx = channel_idx * n_stats * (level + 1)
                features[sample_idx, start_idx : start_idx + len(coeff_features)] = (
                    coeff_features
                )

        return features

    @staticmethod
    def extract_all_features(X, sfreq=250):
        """Combine all feature extraction methods"""
        time_features = FeatureExtractor.extract_time_domain_features(X)
        freq_features = FeatureExtractor.extract_frequency_domain_features(X, sfreq)
        wavelet_features = FeatureExtractor.extract_wavelet_features(X)

        # Verify shapes match
        assert (
            time_features.shape[0]
            == freq_features.shape[0]
            == wavelet_features.shape[0]
        ), f"Feature count mismatch: time={time_features.shape[0]}, freq={freq_features.shape[0]}, wavelet={wavelet_features.shape[0]}"

        return np.concatenate([time_features, freq_features, wavelet_features], axis=1)
