"""
keras_nn.py — sklearn-compatible wrapper for Keras Sequential Neural Network.

Provides a clean interface so that the rest of the pipeline (Tuner, Trainer,
Evaluator) can treat the Neural Network identically to any sklearn estimator.

Key features:
    - Implements fit(), predict(), predict_proba(), get_params(), set_params()
    - Stores architecture hyperparameters for reconstruction
    - Uses joblib-compatible save/load via custom __getstate__/__setstate__
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warnings

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

logger = logging.getLogger(__name__)


class KerasNN(BaseEstimator, ClassifierMixin):
    """A feed-forward neural network that behaves like an sklearn estimator.

    Parameters
    ----------
    layers : list[int]
        Number of neurons in each hidden layer (e.g. [30, 20, 10, 5]).
        The output layer (1 neuron, sigmoid) is always appended automatically.
    dropout : float
        Dropout rate applied after each hidden layer (0.0 = no dropout).
    learning_rate : float
        Adam optimizer learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    activation : str
        Activation function for hidden layers ('relu' or 'tanh').
    random_state : int
        Random seed for reproducibility.
    verbose : int
        Keras training verbosity (0 = silent, 1 = progress bar).
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 35,
        batch_size: int = 250,
        activation: str = "relu",
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.layers = layers or [30, 20, 10, 5]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.random_state = random_state
        self.verbose = verbose
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self._fitted: bool = False

    # ──────────────────────────────────────────────
    # Build & compile
    # ──────────────────────────────────────────────

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build the Sequential model with the configured architecture."""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for i, n_units in enumerate(self.layers):
            model.add(tf.keras.layers.Dense(n_units, activation=self.activation))
            if self.dropout > 0.0:
                model.add(tf.keras.layers.Dropout(self.dropout))

        # Output layer — binary classification
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ──────────────────────────────────────────────
    # sklearn API
    # ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        **kwargs: Any,
    ) -> "KerasNN":
        """Fit the neural network to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        validation_split : float
            Fraction of training data to use for validation.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        self.model = self._build_model(X_arr.shape[1])

        logger.info(
            "Training KerasNN (%d params) on %d samples for %d epochs…",
            self.model.count_params(),
            len(X_arr),
            self.epochs,
        )

        hist = self.model.fit(
            X_arr,
            y_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=self.verbose,
            **kwargs,
        )
        self.history = hist.history
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        probs = self.model.predict(np.asarray(X, dtype=np.float32), verbose=0)
        return (probs > 0.5).astype(np.int32).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (n_samples, 2) — sklearn convention.

        Column 0 = P(class=0), Column 1 = P(class=1).
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        p1 = self.model.predict(np.asarray(X, dtype=np.float32), verbose=0).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return init parameters — required for sklearn compatibility."""
        return {
            "layers": self.layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "activation": self.activation,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "KerasNN":
        """Set init parameters — required for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    # ──────────────────────────────────────────────
    # Serialisation (for joblib)
    # ──────────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        """Custom serialisation — save model weights to temp buffer."""
        state = self.__dict__.copy()
        if self.model is not None:
            # Save Keras model to a temporary file, read bytes
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
                self.model.save(tmp.name, save_format="keras")
                with open(tmp.name, "rb") as f:
                    state["_keras_bytes"] = f.read()
        state["model"] = None  # Don't pickle the Keras model directly
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom deserialisation — restore model weights from bytes."""
        keras_bytes = state.pop("_keras_bytes", None)
        self.__dict__.update(state)
        self.model = None
        self._fitted = False

        if keras_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
                with open(tmp.name, "wb") as f:
                    f.write(keras_bytes)
                self.model = tf.keras.models.load_model(tmp.name)
            self._fitted = True

    # ──────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────

    def summary(self) -> str:
        """Print model architecture summary."""
        if self.model is None:
            return "Model not built yet."
        string_list: List[str] = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def __repr__(self) -> str:
        layers_str = "→".join(str(l) for l in (self.layers or []))
        return (
            f"KerasNN(layers=[{layers_str}→1], "
            f"dropout={self.dropout}, lr={self.learning_rate}, "
            f"epochs={self.epochs}, batch={self.batch_size})"
        )