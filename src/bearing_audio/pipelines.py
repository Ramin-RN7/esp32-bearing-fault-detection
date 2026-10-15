"""The thesis classifiers, trained exactly as in the archived scripts, plus a deliberately simple baseline."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42


class A8GMM:
    """A8 (a8_bandpass_1_5k.py): z-score -> PCA(4) -> 3-component diagonal GMM on healthy training windows.

    A window is a fault when its log-likelihood is below the threshold that maximises fault F1 on the validation
    windows (200 percentile candidates). A recording is a fault when its mean window log-likelihood is below the
    threshold (a35a_aggregate_A8_GMM_per_file.py).
    """
    name, features = "A8 GMM", "a8"

    def fit(self, X, y, X_val, y_val):
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=4, random_state=SEED).fit(self.scaler.transform(X))
        self.gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=SEED).fit(self._z(X)[y == 0])
        scores, best = self.score(X_val), -1.0
        for t in np.percentile(scores, np.linspace(0.5, 99.5, 200)):
            f1 = f1_score(y_val, (scores < t).astype(int), zero_division=0)
            if f1 > best:
                best, self.threshold = f1, t
        return self

    def _z(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def score(self, X):
        return self.gmm.score_samples(self._z(X))

    def predict_windows(self, X):
        return (self.score(X) < self.threshold).astype(int)

    def predict_recording(self, X):
        return int(np.mean(self.score(X)) < self.threshold)


class A16MLP:
    """A16 (a16_env_log_sr8k_pca8_minmax.py): MinMax -> PCA(8) -> MLP(10 ReLU units, Adam, 500 iterations).

    A recording is a fault when the mean window probability is >= 0.5 (a35b_aggregate_A16_MLP_per_file.py).
    """
    name, features = "A16 MLP", "a16"

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler = MinMaxScaler().fit(X)
        self.pca = PCA(n_components=8, random_state=SEED).fit(self.scaler.transform(X))
        self.mlp = MLPClassifier(hidden_layer_sizes=(10,), activation="relu", solver="adam", max_iter=500,
                                 random_state=SEED).fit(self._z(X), y)
        return self

    def _z(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def proba(self, X):
        return self.mlp.predict_proba(self._z(X))[:, 1]

    def predict_windows(self, X):
        return (self.proba(X) >= 0.5).astype(int)

    def predict_recording(self, X):
        return int(self.proba(X).mean() >= 0.5)


class RmsKurtosisBaseline:
    """Sanity baseline: class-balanced logistic regression on two features of the A8 window (RMS, kurtosis)."""
    name, features = "Baseline: RMS + kurtosis", "a8"
    columns = [2, 6]

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler = StandardScaler().fit(X[:, self.columns])
        self.lr = LogisticRegression(class_weight="balanced").fit(self.scaler.transform(X[:, self.columns]), y)
        return self

    def proba(self, X):
        return self.lr.predict_proba(self.scaler.transform(X[:, self.columns]))[:, 1]

    def predict_windows(self, X):
        return (self.proba(X) >= 0.5).astype(int)

    def predict_recording(self, X):
        return int(self.proba(X).mean() >= 0.5)


PIPELINES = {"a8_gmm": A8GMM, "a16_mlp": A16MLP, "baseline_rms_kurtosis": RmsKurtosisBaseline}
