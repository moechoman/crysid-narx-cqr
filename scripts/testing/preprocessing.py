import pandas as pd
import numpy as np

# -----------------------------
# Classification Feature Extraction
# -----------------------------
# Columns used to compute summary statistics for product classification
_CLASSIFICATION_BASE_COLS = [
    'd10', 'd50', 'd90',
    'c', 'T_PM', 'T_TM',
    'mf_PM', 'mf_TM', 'Q_g', 'w_crystal',
    'c_in', 'T_PM_in', 'T_TM_in'
]

CLASSIFICATION_FEATURE_NAMES = []
for col in _CLASSIFICATION_BASE_COLS:
    CLASSIFICATION_FEATURE_NAMES += [
        f"{col}_mean", f"{col}_std", f"{col}_smooth"
    ]


def extract_summary_features(file_path: str) -> np.ndarray:
    """
    Compute summary statistics (mean, std, smoothness) for each relevant variable
    in the trajectory file for product classification.

    Returns:
        feature_vector: 1D numpy array matching CLASSIFICATION_FEATURE_NAMES order.
    """
    df = pd.read_csv(file_path, sep='\t')
    feats = []
    for col in _CLASSIFICATION_BASE_COLS:
        if col in df.columns:
            arr = df[col].dropna().values
            mean = arr.mean() if arr.size else 0.0
            std  = arr.std()  if arr.size else 0.0
            smooth = np.mean(np.abs(np.diff(arr))) if arr.size>1 else 0.0
        else:
            mean = std = smooth = 0.0
        feats += [mean, std, smooth]
    return np.array(feats)

# -----------------------------
# NARX Data Preparation
# -----------------------------

CRYSTAL_COLS = ['d10', 'd50', 'd90']


def _iqr_clip_array(arr: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Clip array to [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    if arr.size == 0:
        return arr
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return np.clip(arr, q1 - factor*iqr, q3 + factor*iqr)


def load_and_prepare_narx_data(
    file_path: str,
    lag: int,
    input_cols: list,
    output_cols: list
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Load a single trajectory, clip crystal-size outliers, and build NARX sequences.

    Args:
        file_path: path to .txt data file (tab-delimited)
        lag: number of past timesteps to use
        input_cols: list of input variable names
        output_cols: list of output variable names

    Returns:
        X_seq: ndarray of shape (n_samples, lag*(n_outputs+n_inputs))
        Y_seq: ndarray of shape (n_samples, n_outputs)
        Y_full: ndarray of all raw outputs shape (n_timesteps, n_outputs)
    """
    df = pd.read_csv(file_path, sep='\t')
    # ensure required columns
    if not all(c in df.columns for c in input_cols + output_cols):
        return None, None, None
    # clip crystal-size outputs
    for col in CRYSTAL_COLS:
        if col in output_cols:
            df[col] = _iqr_clip_array(df[col].values)
    U = df[input_cols].values
    Y_full = df[output_cols].values

    X_list, Y_list = [], []
    for t in range(lag, len(df)):
        y_hist = Y_full[t-lag:t].flatten()
        u_hist = U[t-lag:t].flatten()
        X_list.append(np.hstack([y_hist, u_hist]))
        Y_list.append(Y_full[t])

    return np.array(X_list), np.array(Y_list), Y_full
