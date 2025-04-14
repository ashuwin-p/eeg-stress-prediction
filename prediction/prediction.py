import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from FeatureExtractor import FeatureExtractor
from DataLoader import EEGDataLoader
from pathlib import Path
from typing import Dict, List, Union
import gc

class ConfigWrapper:
    """Wrapper class to provide the expected interface for EEGDataLoader"""
    def __init__(self, channels: List[str], labels: Dict[str, int]):
        self._channels = channels
        self._labels = labels
    
    def get_labels(self) -> Dict[str, int]:
        return self._labels
    
    def get_channels(self) -> List[str]:
        return self._channels

class EEGStressPredictor:
    CHANNELS = [
        "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9",
        "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
        "TP10", "CP6", "CP2", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"
    ]
    
    def __init__(self, model: object, scaler: object):
        """Initialize predictor with trained model and scaler"""
        self.model = model
        self.scaler = scaler
        self.feature_extractor = FeatureExtractor()
        self.loader = self._initialize_loader()

    def _initialize_loader(self) -> EEGDataLoader:
        """Configure and initialize EEG data loader"""
        print(f"\nConfiguring with {len(self.CHANNELS)} channels")
        config = ConfigWrapper(channels=self.CHANNELS, labels={})
        return EEGDataLoader(config)

    def predict_from_csv(self, csv_path: Union[str, Path]) -> Dict[str, float]:
        """Generate stress predictions from EEG CSV"""
        print(f"\n{' Processing ' + Path(csv_path).name + ' ':=^50}")
        
        # Pipeline execution
        clean_epochs = self._load_and_preprocess(csv_path)
        X = self._extract_features(clean_epochs)
        proba = self._predict_probabilities(X)
        
        self._plot_probabilities(proba)
        return self._format_results(proba)

    def _load_and_preprocess(self, csv_path: Union[str, Path]) -> np.ndarray:
        """Step 1: Data loading and preprocessing"""
        print("\n1. Data Loading and Preprocessing")
        raw_data = self.loader._load_single_file(csv_path)
        clean_data = self.loader._preprocess_eeg(raw_data)
        epochs = self.loader._create_epochs(clean_data)
        clean_epochs = self.loader._remove_bad_epochs(epochs)
        print(f"→ Clean epochs: {len(clean_epochs)}/{len(epochs)} retained")
        del raw_data, clean_data, epochs
        gc.collect()
        return clean_epochs

    def _extract_features(self, epochs: np.ndarray) -> np.ndarray:
        """Step 2: Feature extraction"""
        print("\n2. Feature Extraction")
        X = self.feature_extractor.extract_all_features(epochs)
        print(f"→ Feature matrix: {X.shape} (mean={np.mean(X):.2f} ± {np.std(X):.2f})")
        del epochs
        gc.collect()
        return X

    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Step 3: Prediction"""
        print("\n3. Prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def _format_results(self, proba: np.ndarray) -> Dict[str, float]:
        """Format results dictionary"""
        return {
            "per_epoch_probabilities": proba,
            "mean_stress_probability": np.mean(proba),
            "stress_ratio": np.mean(proba > 0.5),
            "epoch_count": len(proba)
        }

    def _plot_probabilities(self, proba: np.ndarray):
        """Visualization of results"""
        plt.figure(figsize=(10, 4))
        plt.plot(proba, 'b-', alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
        plt.xlabel("Epoch (1-second windows)")
        plt.ylabel("Stress Probability")
        plt.title("Stress Probability Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Direct prediction from a pandas DataFrame"""
        # Save to temp file in memory (or implement full memory-based prediction)
        temp_path = "temp_from_df.csv"
        df.to_csv(temp_path, index=False)
        return self.predict_from_csv(temp_path)


# if __name__ == "__main__":
#     print("\n" + " EEG Stress Prediction Pipeline ".center(50, '='))
    
#     # Load model and scaler
#     model = joblib.load("best_xgboost_model.pkl")
#     scaler = joblib.load("scaler.pkl")

#     # Pass the loaded objects directly
#     predictor = EEGStressPredictor(model, scaler)

#     results = predictor.predict_from_csv(r"U:\EEG-MUSI\data\converted_full_info\subject_211_pre.csv")
    
#     print("\n" + " Results ".center(50, '-'))
#     print(f"• Mean Stress Probability: {results['mean_stress_probability']:.1%}")
#     print(f"• Stress Epochs Ratio: {results['stress_ratio']:.1%}")
#     print(f"• Valid Epochs Analyzed: {results['epoch_count']}")
#     print("="*50)
