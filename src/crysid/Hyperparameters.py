# %%

"""
hyperparameter_tuning.py

Use Optuna to find optimal hyperparameters for the NARX-ANN weighted-MSE training
pipeline defined in Clean_Source_Code.py. This script will tune:

- lag (history length)
- hidden layer topology
- activation function
- optimizer learning rate
- batch size
- dropout rate
- early-stopping patience
- per-output weights for the multi-output MSE loss

Requirements:
    pandas, numpy, scikit-learn, torch, optuna
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import optuna

# ──────────────────────────────────────────────────────────────────────────────
# Global configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_FOLDER = "full_dataset"
PRODUCT_CSV = "Product1.csv"  # adjust for product 1 or 2
INPUT_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
OUTPUT_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Precomputed MSE thresholds for normalization (one per OUTPUT_COLS)
MSE_THRESHOLDS = [0.010, 4e-08, 1.823e-06, 1.438e-06, 1.566e-06, 0.0100]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing utilities
# ──────────────────────────────────────────────────────────────────────────────

def iqr_clip(arr: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Clip values in `arr` to [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    q1, q3 = np.percentile(arr, [25, 75])
    iqr   = q3 - q1
    return np.clip(arr, q1 - factor*iqr, q3 + factor*iqr)

def load_and_prepare_narx_data(path: str, lag: int):
    """
    Read one trajectory from `path`, apply IQR clipping to outputs,
    and build NARX sequences of length `lag`.
    Returns:
        X: (n_samples, lag*(n_in+n_out)) feature array
        Y: (n_samples, n_out)           target array
    """
    df = pd.read_csv(path, sep="\t")
    # ensure all columns present
    if not all(col in df.columns for col in INPUT_COLS + OUTPUT_COLS):
        return None, None

    # clip outliers on each output channel
    for col in OUTPUT_COLS:
        df[col] = iqr_clip(df[col].values)

    U = df[INPUT_COLS].values
    Y = df[OUTPUT_COLS].values

    X_list, Y_list = [], []
    for t in range(lag, len(df)):
        y_hist = Y[t-lag:t].reshape(-1)
        u_hist = U[t-lag:t].reshape(-1)
        X_list.append(np.hstack([y_hist, u_hist]))
        Y_list.append(Y[t])
    return np.array(X_list), np.array(Y_list)

def build_dataset(file_list, lag: int):
    """
    Aggregate multiple trajectories into a single feature/target dataset.
    """
    Xs, Ys = [], []
    for fname in file_list:
        filepath = os.path.join(DATA_FOLDER, fname)
        if not os.path.exists(filepath):
            continue
        X, Y = load_and_prepare_narx_data(filepath, lag)
        if X is not None:
            Xs.append(X); Ys.append(Y)
    if not Xs:
        return np.zeros((0,)), np.zeros((0,))
    return np.vstack(Xs), np.vstack(Ys)

# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class NARXANN(nn.Module):
    """
    Feed-forward NARX network: takes lagged y and u histories as input.
    """
    def __init__(self, input_dim, output_dim, hidden_layers, activation_cls, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(in_dim, h),
                activation_cls(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning objective
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial):
    # Suggest hyperparameters
    lag = trial.suggest_int("lag", 1, 10)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = [
        trial.suggest_int(f"n_units_l{i}", 32, 512, log=True)
        for i in range(n_layers)
    ]
    dropout   = trial.suggest_float("dropout", 0.0, 0.5)
    lr        = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size= trial.suggest_categorical("batch_size", [32, 64, 128])
    patience  = trial.suggest_int("patience", 5, 20)

    # One weight per output channel
    weight_vec = [
        trial.suggest_float(f"w_{i}", 0.1, 10.0, log=True)
        for i in range(len(OUTPUT_COLS))
    ]
    weight_tensor = torch.tensor(weight_vec, device=DEVICE)


    # Load file list
    df = pd.read_csv(PRODUCT_CSV)
    all_files = df["source_file"].unique().tolist()
    # train/val/test split
    train_val, test_files = train_test_split(all_files, test_size=0.15, random_state=42)
    train_files, val_files = train_test_split(
        train_val, test_size=len(test_files)/len(train_val), random_state=42
    )

    # Build datasets
    X_tr, Y_tr = build_dataset(train_files, lag)
    X_va, Y_va = build_dataset(val_files,   lag)
    X_te, Y_te = build_dataset(test_files,  lag)

    # Standardize
    sx = StandardScaler().fit(X_tr); X_tr_s = sx.transform(X_tr)
    sy = StandardScaler().fit(Y_tr); Y_tr_s = sy.transform(Y_tr)
    X_va_s = sx.transform(X_va);     Y_va_s = sy.transform(Y_va)
    X_te_s = sx.transform(X_te);     Y_te_s = sy.transform(Y_te)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s).float(), torch.from_numpy(Y_tr_s).float()),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_s).float(), torch.from_numpy(Y_va_s).float()),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te_s).float(), torch.from_numpy(Y_te_s).float()),
        batch_size=batch_size, shuffle=False
    )

    # Initialize model, optimizer, loss
    model = NARXANN(
        input_dim = X_tr_s.shape[1],
        output_dim= Y_tr_s.shape[1],
        hidden_layers=hidden_sizes,
        activation_cls=nn.ReLU,
        dropout=dropout
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def weighted_mse(pred, tgt):
        return torch.mean((pred - tgt) ** 2 * weight_tensor)

    # Training with early stopping
    best_val = float("inf")
    patience_cnt = 0
    for epoch in range(100):  # hard cap on epochs
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(weighted_mse(model(xb), yb).item() * xb.size(0))
        mean_val = np.sum(val_losses) / len(val_loader.dataset)

        # early-stop check
        if mean_val < best_val:
            best_val = mean_val
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # load best model
    model.load_state_dict(best_state)

    # Test-time evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = sy.inverse_transform(np.vstack(preds))
    y_true = sy.inverse_transform(np.vstack(trues))

    # Compute normalized MSE score
    score = 0.0
    for i, thr in enumerate(MSE_THRESHOLDS):
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        score += mse_i / thr

    return score

# ──────────────────────────────────────────────────────────────────────────────
# Run the hyperparameter search
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=3_600)  # up to 1h or 50 trials

    print("Best hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
    print(f"Best normalized MSE score: {study.best_value:.4f}")

# %%
