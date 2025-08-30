# %%
# Main.py
# This script performs classification and NARX ANN prediction for a given test file.
# It loads the necessary models, scalers, and data, then computes predictions and metrics.

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import preprocessing utilities
from preprocessing import extract_summary_features, load_and_prepare_narx_data, CRYSTAL_COLS, _iqr_clip_array

# -----------------------------
# Configuration
# -----------------------------
CLASSIFIER_PATH = 'product_classifier.pkl'
MODEL_PATHS = {
    1: 'narxann_product1.pth',
    2: 'narxann_product2.pth'
}
SCALER_X_PATHS = {
    1: 'scaler_x_product1.save',
    2: 'scaler_x_product2.save'
}
SCALER_Y_PATHS = {
    1: 'scaler_y_product1.save',
    2: 'scaler_y_product2.save'
}

# NARX model configuration per product
PRODUCT_CONFIGS = {
    1: {'lag': 5, 'hidden_layers': [128,64,64], 'dropout': 0.035490764806863494},
    2: {'lag': 5, 'hidden_layers': [128,64,64], 'dropout': 0.035490764806863494}
}

INPUT_COLS = ['mf_PM','mf_TM','Q_g','w_crystal','c_in','T_PM_in','T_TM_in']
OUTPUT_COLS = ['T_PM','c','d10','d50','d90','T_TM']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# NARX ANN definition (must match training)
# -----------------------------
class NARXANN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)  # matches checkpoint keys
    def forward(self, x):
        return self.model(x)

# -----------------------------
# Main routine
# -----------------------------
def main():
    # specify test filename manually
    test_file = 'file_12738.txt'  # <-- edit this to your test file path

    if not os.path.isfile(test_file):
        print(f"File not found: {test_file}")
        sys.exit(1)

    # 1) Classification
    feats = extract_summary_features(test_file)
    clf = joblib.load(CLASSIFIER_PATH)
    pid = int(clf.predict(feats.reshape(1,-1))[0])
    # Validate classification
    if pid not in (1, 2):
        print("Batch doesn't belong to Product 1 or Product 2; it may be corrupted.")
        sys.exit(1)
    print(f"Classified as Product {pid}")

    # config for selected product
    cfg = PRODUCT_CONFIGS[pid]
    LAG = cfg['lag']
    hidden = cfg['hidden_layers']
    dropout = cfg['dropout']

    # 2) NARX data preparation
    X_seq, Y_seq, Y_full = load_and_prepare_narx_data(
        test_file, LAG, INPUT_COLS, OUTPUT_COLS
    )
    if X_seq is None:
        print("Required columns missing in data")
        sys.exit(1)

    # 3) Load scalers & model
    sx = joblib.load(SCALER_X_PATHS[pid])
    sy = joblib.load(SCALER_Y_PATHS[pid])
    model = NARXANN(
        input_dim = X_seq.shape[1],
        output_dim = len(OUTPUT_COLS),
        hidden_layers = hidden,
        dropout = dropout
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATHS[pid], map_location=DEVICE))
    model.eval()

    # 4) Closed-loop prediction
    X_s = sx.transform(X_seq)
    with torch.no_grad():
        y_s = model(torch.tensor(X_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    y_pred_closed = sy.inverse_transform(y_s)

    # assemble full closed-loop trajectory
    Yp_full = np.vstack([Y_full[:LAG], y_pred_closed])

    # 5) Open-loop prediction
    df = pd.read_csv(test_file, sep='\t')
    # clip crystal outputs
    for col in CRYSTAL_COLS:
        if col in df.columns:
            df[col] = _iqr_clip_array(df[col].values)
    y_full = df[OUTPUT_COLS].values
    u_seq = df[INPUT_COLS].values

    deq = deque(y_full[:LAG], maxlen=LAG)
    ol = []
    for t in range(LAG, len(df)):
        y_hist = np.hstack(deq)
        u_hist = u_seq[t-LAG:t].flatten()
        x_raw = np.hstack([y_hist, u_hist])[None, :]
        x_s = sx.transform(x_raw)
        with torch.no_grad():
            p_s = model(torch.tensor(x_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        p = sy.inverse_transform(p_s)[0]
        ol.append(p)
        deq.append(p)
    Y_ol_full = np.vstack([y_full[:LAG], np.vstack(ol)])

    # 6) Compute and Display Metrics ---
    print("\nMetrics for Open‐Loop and Closed‐Loop Predictions:")
    print(f"\n{'Open-Loop Metrics:':<32}{'  |  '}{'  Closed-Loop Metrics:':<32}")
    print(f"{'State':<8}{'MSE':>12}{'MAE':>12}{'  |  '}{'MSE':>12}{'MAE':>12}")
    for i, name in enumerate(OUTPUT_COLS):
        mse_open  = mean_squared_error(Y_seq[:, i], ol[i][0] if False else np.vstack(ol)[:, i])
        mae_open  = mean_absolute_error(Y_seq[:, i], np.vstack(ol)[:, i])
        mse_close = mean_squared_error(Y_seq[:, i], y_pred_closed[:, i])
        mae_close = mean_absolute_error(Y_seq[:, i], y_pred_closed[:, i])
        print(f"{name:<8}{mse_open:12.4e}{mae_open:12.4e}  |  {mse_close:12.4e}{mae_close:12.4e}")

    # --- Plot Alignment of Predictions vs True Trajectory ---
    t = np.arange(len(y_full))
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.ravel()
    for i, name in enumerate(OUTPUT_COLS):
        ax = axes[i]
        ax.plot(t, y_full[:, i],    color='blue',  label='True')
        ax.plot(t, Yp_full[:, i], color='red',   label='Closed-Loop')
        ax.plot(t, Y_ol_full[:, i],     '--',  color='green', label='Open-Loop')
        ax.set_title(name)
        ax.legend()
    plt.suptitle(f"Open-Loop vs Closed-Loop Predictions", y=1.02)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()

# %%
