"""
Model zoo for the energy-theft classification task (slide 5 / slide 6):
  - Random Forest        (tabular engineered features)
  - XGBoost               (tabular engineered features)
  - LSTM                  (raw daily consumption sequence)

Each train_* function returns a fitted model plus predicted probabilities
on the held-out test split, so evaluate.py can score them uniformly.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ---------------------------------------------------------------- Tabular --

def train_random_forest(X_train, y_train, X_test, random_state=42):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return clf, proba


def train_xgboost(X_train, y_train, X_test, random_state=42):
    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    scale_pos_weight = neg / pos

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return clf, proba


# -------------------------------------------------------------------- LSTM --

def _build_lstm(n_timesteps, n_features=1):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(n_timesteps, n_features)),
            layers.Masking(mask_value=0.0),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def _downsample_sequence(series_matrix, target_len=180):
    """Average-pool a long daily series down to target_len steps so the
    LSTM trains quickly; keeps the seasonal/theft shape intact."""
    n_days = series_matrix.shape[1]
    if n_days <= target_len:
        return series_matrix
    factor = n_days // target_len
    trimmed = series_matrix[:, : factor * target_len]
    reshaped = trimmed.reshape(series_matrix.shape[0], target_len, factor)
    return reshaped.mean(axis=2)


def train_lstm(seq_train, y_train, seq_test, epochs=15, batch_size=32, random_state=42):
    import tensorflow as tf

    tf.random.set_seed(random_state)

    seq_train_ds = _downsample_sequence(seq_train)
    seq_test_ds = _downsample_sequence(seq_test)

    scaler = StandardScaler()
    n_train = seq_train_ds.shape[0]
    flat_train = scaler.fit_transform(seq_train_ds.reshape(-1, 1)).reshape(seq_train_ds.shape)
    flat_test = scaler.transform(seq_test_ds.reshape(-1, 1)).reshape(seq_test_ds.shape)

    X_train = flat_train[..., np.newaxis]
    X_test = flat_test[..., np.newaxis]

    model = _build_lstm(n_timesteps=X_train.shape[1])

    pos = max(y_train.sum(), 1)
    neg = max(len(y_train) - y_train.sum(), 1)
    class_weight = {0: 1.0, 1: neg / pos}

    early_stop = __import__("tensorflow").keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=0,
        callbacks=[early_stop],
    )
    proba = model.predict(X_test, verbose=0).ravel()
    return model, proba
