# %% 
# Cell 1: Setup, Imports, Data Loading & Inspection

# Import necessary libraries:
# - os for file path operations
# - pandas for table-like data handling
# - numpy for numerical computations
# - matplotlib.pyplot for plotting
# - random for sampling
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = BASE_DIR / "data"

# Specify where batch data is stored (relative path)
data_dir = DATA_DIR / "full_dataset"
file_path = data_dir

# Gather all file paths in the data directory
data_files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f))
]
print(f"Found {len(data_files)} files in '{data_dir}'")

# Function to load one batch file:
# - Automatically detect delimiter
# - Tag the DataFrame with its source filename
def load_batch(file_path):
    df = pd.read_csv(file_path, sep=None, engine='python')
    df['source_file'] = os.path.basename(file_path)
    return df

# Read every batch file into a list of DataFrames
dfs = [load_batch(fp) for fp in data_files]

# Preview the first few rows of the first batch
print(dfs[0].head())

# Show structure and data types of the first batch
print("\nData info:")
dfs[0].info()

# %%
# Cell 2: Plot Time-Series Dynamics for a Random Sample of Batches

# Pick three random batches to visualize
selected = random.sample(dfs, 3)

# Identify numeric measurement columns (assumes same schema across batches)
num_cols = selected[0].select_dtypes(include=[np.number]).columns.tolist()

# Set up a vertical subplot for each variable
n_vars = len(num_cols)
fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)

# For each variable, overlay line and scatter plots from the three batches
for i, col in enumerate(num_cols):
    ax = axes[i]
    for df in selected:
        src = df['source_file'].iloc[0]
        # Draw a line connecting consecutive samples
        ax.plot(df.index, df[col], '-', alpha=0.8, label=src)
        # Overlay raw data points
        ax.scatter(df.index, df[col], s=10, alpha=0.4)
    ax.set_ylabel(col)
    ax.legend(loc='upper right')

# Add a label for the shared x-axis
axes[-1].set_xlabel("Time step")

plt.tight_layout()
plt.show()

# %%
# Cell 3: Outlier Removal and Rolling-Median Smoothing for Particle Sizes

def iqr_clip_series(series, k=1.5):
    """
    Clip values in a pandas Series to the [Q1 − k·IQR, Q3 + k·IQR] range.
    This helps mitigate extreme outliers based on the interquartile range.
    """
    arr = series.values
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    clipped = np.clip(arr, lower, upper)
    return pd.Series(clipped, index=series.index)

def smooth(series, window=11):
    """
    Apply a centered rolling-median filter to the Series.
    Window length defaults to 11, with at least 1 period required.
    """
    return series.rolling(window=window, center=True, min_periods=1).median()

# Apply outlier clipping and smoothing to each batch DataFrame
dfs_smoothed = []
for df in dfs:
    df2 = df.copy()
    for col in ['d10', 'd50', 'd90']:
        if col in df2.columns:
            # Step 1: Clip extreme values based on IQR
            clipped = iqr_clip_series(df2[col], k=1.5)
            # Step 2: Smooth clipped data with rolling median
            df2[col] = smooth(clipped, window=21)
    dfs_smoothed.append(df2)

# Re-plot the same three randomly selected runs on smoothed data
selected = random.sample(dfs_smoothed, 3)
num_cols = selected[0].select_dtypes(include=[np.number]).columns.tolist()
n = len(num_cols)
fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

for i, col in enumerate(num_cols):
    ax = axes[i]
    for df in selected:
        src = df['source_file'].iloc[0]
        # Draw smoothed time series line and raw points
        ax.plot(df.index, df[col], '-', alpha=0.8, label=src)
        ax.scatter(df.index, df[col], s=10, alpha=0.4)
    ax.set_ylabel(col)
    ax.legend(loc='upper right')

# Add x-axis label for all subplots
axes[-1].set_xlabel("Time step")

plt.tight_layout()
plt.show()

# %%
# Cell 4: Feature Extraction and PCA Comparison

# 1. Imports required for feature extraction and PCA
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
from mpl_toolkits.mplot3d import Axes3D

# 2. Define data directory and expected measurement columns
data_dir = DATA_DIR / "full_dataset"
column_names = [
    "c", "T_PM", "d50", "d90", "d10", "T_TM",
    "mf_PM", "mf_TM", "Q_g", "w_crystal",
    "c_in", "T_PM_in", "T_TM_in"
]

# 3. Extract summary statistics (mean, std, smoothness) from each file
features = []
file_names = []
for fname in os.listdir(data_dir):
    if fname.endswith('.txt'):
        df = pd.read_csv(os.path.join(data_dir, fname), sep=None, engine='python')
        stats = {}
        # Particle size stats
        for col in ['d10', 'd50', 'd90']:
            if col in df:
                s = df[col].dropna()
                stats[f'{col}_mean']   = s.mean()
                stats[f'{col}_std']    = s.std()
                stats[f'{col}_smooth'] = np.mean(np.abs(np.diff(s)))
        # State and control stats
        extras = ['c','T_PM','T_TM','mf_PM','mf_TM','Q_g','w_crystal','c_in','T_PM_in','T_TM_in']
        for col in extras:
            if col in df:
                s = df[col].dropna()
                stats[f'{col}_mean']   = s.mean()
                stats[f'{col}_std']    = s.std()
                stats[f'{col}_smooth'] = np.mean(np.abs(np.diff(s)))
        features.append(stats)
        file_names.append(fname)

features_df = pd.DataFrame(features, index=file_names)

# 4. PCA helper function
def run_pca(df, n_components=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)
    return pca, pcs

# 5. Perform PCA on three feature sets:
#    a) Means only
mean_cols = [c for c in features_df.columns if c.endswith('_mean')]
pca1, pcs1 = run_pca(features_df[mean_cols])
#    b) Means + standard deviations
mean_std_cols = [c for c in features_df.columns if c.endswith(('_mean','_std'))]
pca2, pcs2 = run_pca(features_df[mean_std_cols])
#    c) All statistics
pca3, pcs3 = run_pca(features_df)

# 6. Print explained variance ratios for each PCA variant
print('Explained variance (means):', pca1.explained_variance_ratio_)
print('Explained variance (means+std):', pca2.explained_variance_ratio_)
print('Explained variance (all stats):', pca3.explained_variance_ratio_)

# 7. Convert PCA results into DataFrames for plotting
pcs1_df = pd.DataFrame(pcs1, index=file_names, columns=['PC1','PC2','PC3'])
pcs2_df = pd.DataFrame(pcs2, index=file_names, columns=['PC1','PC2','PC3'])
pcs3_df = pd.DataFrame(pcs3, index=file_names, columns=['PC1','PC2','PC3'])

# 8. 2D Visualization: PC1 vs PC2 for each PCA variant
fig, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True, sharey=True)
for ax, (pcs_df, title) in zip(axes, 
        [(pcs1_df, 'Means only'),
         (pcs2_df, 'Means + Std'),
         (pcs3_df, 'All stats')]):
    ax.scatter(pcs_df['PC1'], pcs_df['PC2'], s=50, alpha=0.8)
    ax.set_title(f'2D PCA: {title}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# 9. 3D Visualization: PC1, PC2, PC3 for each PCA variant
fig = plt.figure(figsize=(18,5))
for i, (pcs_df, title) in enumerate(
        [(pcs1_df, 'Means only'),
         (pcs2_df, 'Means + Std'),
         (pcs3_df, 'All stats')], start=1):
    ax = fig.add_subplot(1, 3, i, projection='3d')
    ax.scatter(pcs_df['PC1'], pcs_df['PC2'], pcs_df['PC3'], s=30, alpha=0.7)
    ax.set_title(f'3D PCA: {title}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.view_init(elev=20, azim=30)
plt.tight_layout()
plt.show()

# %%
# Cell 5: HDBSCAN Clustering on 3D PCA Embedding

import hdbscan
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Extract the 3D coordinates from the "All stats" PCA DataFrame
X_hdb = pcs3_df[['PC1', 'PC2', 'PC3']].values

# Configure and run HDBSCAN for density-based clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=40,        # smallest allowed cluster
    min_samples=15,             # how conservative the clustering is
    cluster_selection_method='leaf'
)
labels = clusterer.fit_predict(X_hdb)  # -1 marks noise points

# Summarize cluster counts and noise
unique, counts = np.unique(labels, return_counts=True)
n_clusters = np.sum(unique >= 0)
noise_count = int(counts[unique == -1][0]) if -1 in unique else 0

print(f"Detected clusters: {n_clusters}")
for lbl, cnt in zip(unique, counts):
    if lbl == -1:
        print(f"Noise (corrupted batches): {cnt}")
    else:
        print(f"Cluster {lbl}: {cnt} batches")

# Plot the clustering result in 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
palette = plt.cm.get_cmap('tab10', len(unique))

for lbl in unique:
    mask = (labels == lbl)
    if lbl == -1:
        color = 'lightgrey'
        label_name = 'noise'
    else:
        color = palette(lbl % 10)
        label_name = f'cluster {lbl}'
    ax.scatter(
        X_hdb[mask, 0],
        X_hdb[mask, 1],
        X_hdb[mask, 2],
        c=[color],
        label=label_name,
        s=50,
        alpha=0.8
    )

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('HDBSCAN Clusters on All-Stats PCA')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

# %%
# Cell 6: IsolationForest + KMeans Clustering on PCA Embedding

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Use the 3-component PCA DataFrame (pcs3_df) with PC1, PC2, PC3 columns
X_if_km = pcs3_df[['PC1', 'PC2', 'PC3']].values

# Detect outliers using Isolation Forest (-1 indicates outlier, 1 indicates inlier)
iso = IsolationForest(contamination=0.1, random_state=42)
flags = iso.fit_predict(X_if_km)

# Initialize all labels as -1 (corrupted/outlier)
if_km_labels = np.full(flags.shape, -1)

# Select inliers for clustering
inlier_idx = np.where(flags == 1)[0]
if inlier_idx.size == 0:
    raise RuntimeError("No inliers detected by IsolationForest.")
inlier_data = X_if_km[inlier_idx]

# Cluster the inliers into two groups with KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_ids = kmeans.fit_predict(inlier_data)

# Map cluster IDs back to the full dataset
for idx, lbl in zip(inlier_idx, cluster_ids):
    if_km_labels[idx] = lbl

# Summarize clusters and noise
unique, counts = np.unique(if_km_labels, return_counts=True)
n_clusters = np.sum(unique >= 0)
noise_count = int(counts[unique == -1][0]) if -1 in unique else 0

print(f"Detected IF+KMeans clusters: {n_clusters}")
for lbl, cnt in zip(unique, counts):
    if lbl == -1:
        print(f"Corrupted batches (noise): {cnt}")
    else:
        print(f"Cluster {lbl}: {cnt}")

# 3D visualization of the clustering result
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
palette = plt.cm.get_cmap('tab10', n_clusters)

for lbl in unique:
    mask = (if_km_labels == lbl)
    color = 'lightgrey' if lbl == -1 else palette(lbl % 10)
    name = 'noise' if lbl == -1 else f'cluster {lbl}'
    ax.scatter(
        X_if_km[mask, 0],
        X_if_km[mask, 1],
        X_if_km[mask, 2],
        c=[color],
        label=name,
        s=50,
        alpha=0.8
    )

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('IsolationForest + KMeans on PCA Embedding')
ax.legend(loc='best')
plt.tight_layout()
plt.show()

# %%
# Cell 7: Save Cluster Assignments and Plot Clean PCA Projections

# Combine extracted features with HDBSCAN cluster labels
cluster_df = features_df.copy()
cluster_df['cluster']     = labels               # -1 = noise, 0/1 = product clusters
cluster_df['source_file'] = cluster_df.index     # restore filename column

# Separate into product 1, product 2, and corrupted batches
prod1   = cluster_df[cluster_df['cluster'] ==  0]
prod2   = cluster_df[cluster_df['cluster'] ==  1]
corrupt = cluster_df[cluster_df['cluster'] == -1]

# Export each group to its own CSV file
prod1.to_csv(DATA_DIR /'Product1.csv', index=False)
prod2.to_csv(DATA_DIR /'Product2.csv', index=False)
corrupt.to_csv(DATA_DIR /'Corrupted_batches.csv', index=False)

print(f"Saved Product1.csv with {len(prod1)} batches.")
print(f"Saved Product2.csv with {len(prod2)} batches.")
print(f"Saved Corrupted_batches.csv with {len(corrupt)} corrupted batches.")

# Prepare boolean index for all non-noise ("clean") batches
clean_idx  = (labels >= 0)

# Define a helper to plot PC1 vs PC2 for two clusters
def plot_pca(ax, pcs_df, title):
    """
    Scatter PC1 vs PC2 for clusters 0 and 1.
    Only includes points where cluster label >= 0.
    """
    for cluster_id, color in zip([0, 1], ['C0', 'C1']):
        mask = (labels == cluster_id) & clean_idx
        ax.scatter(
            pcs_df.loc[mask, 'PC1'],
            pcs_df.loc[mask, 'PC2'],
            s=50, alpha=0.8,
            c=color,
            label=f'Cluster {cluster_id}'
        )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend()

# Create a 1×3 grid showing the 2D PCA for each feature set
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
plot_pca(axes[0], pcs1_df, 'Means only PCA')
plot_pca(axes[1], pcs2_df, 'Means + Std PCA')
plot_pca(axes[2], pcs3_df, 'All stats PCA')

plt.tight_layout()
plt.show()

# %% 
# Cell 8: Classification model Building, training, evaluation, and saving

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df1 = pd.read_csv(DATA_DIR /'Product1.csv'); df1['label'] = 1
df2 = pd.read_csv(DATA_DIR /'Product2.csv'); df2['label'] = 2

# combine, shuffle, and drop the HDBSCAN 'cluster' column from features
df_clf = pd.concat([df1, df2], ignore_index=True) \
           .sample(frac=1, random_state=42)
feature_cols = [c for c in df_clf.columns 
                if c not in ['source_file','label','cluster']]
X_clf = df_clf[feature_cols]
y_clf = df_clf['label']

# split for quick validation
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

# fit classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# evaluate
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy: {acc:.3f}")

# save final model
joblib.dump(clf,DATA_DIR / 'product_classifier.pkl')
print("✔️  Trained and saved product classifier (product_classifier.pkl)")

# %%
# Cell 9: NARX_ANN Training, Evaluation, and Saving for Selected Product

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ──────────────────────────────────────────────
# 0. Select product-specific configuration
# ──────────────────────────────────────────────
product_configs = {
    1: {
        'csv':           'Product1.csv',
        'LAG':           5,
        'HIDDEN_LAYERS': [128, 64, 64],
        'ACTIVATION':    nn.ReLU,
        'OPTIMIZER_FN':  torch.optim.Adam,
        'LEARNING_RATE': 0.000946923696847868,
        'BATCH_SIZE':    64,
        'EPOCHS':        100,
        'PATIENCE':      10,
        'DROPOUT':       0.035490764806863494,
        'WEIGHTS':       [2.35, 2.0, 1.0, 1.0, 1.0, 1.5]
    },
    2: {
        'csv':           'Product2.csv',
        'LAG':           5,
        'HIDDEN_LAYERS': [128, 64, 64],
        'ACTIVATION':    nn.ReLU,
        'OPTIMIZER_FN':  torch.optim.Adam,
        'LEARNING_RATE': 0.000946923696847868,
        'BATCH_SIZE':    64,
        'EPOCHS':        100,
        'PATIENCE':      10,
        'DROPOUT':       0.035490764806863494,
        'WEIGHTS':       [2.35, 2.0, 1.0, 1.0, 1.0, 1.5]
    }
}

# Choose which product to train (1 or 2)
RUN_ID   = 1
cfg      = product_configs[RUN_ID]
prod_csv = cfg['csv']
prod_name = os.path.splitext(prod_csv)[0]  # e.g., "product1"

# Override global constants for this run
LAG            = cfg['LAG']
HIDDEN_LAYERS  = cfg['HIDDEN_LAYERS']
ACTIVATION     = cfg['ACTIVATION']
OPTIMIZER_FN   = cfg['OPTIMIZER_FN']
LEARNING_RATE  = cfg['LEARNING_RATE']
BATCH_SIZE     = cfg['BATCH_SIZE']
EPOCHS         = cfg['EPOCHS']
PATIENCE       = cfg['PATIENCE']
DROPOUT        = cfg['DROPOUT']
WEIGHTS        = cfg['WEIGHTS']

# Define columns used for NARX modeling
INPUT_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
OUTPUT_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
CRYSTAL_COLS = ['d10', 'd50', 'd90']

# Clip outliers in a Series using the IQR rule
def iqr_clip_series_simple(series):
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    return np.clip(series, q1 - 1.5*iqr, q3 + 1.5*iqr)

# Load one batch and build lagged NARX sequences
def load_and_prepare_narx_data(file_path, lag):
    df = pd.read_csv(file_path, sep='\t')
    # Ensure required columns are present
    if not all(col in df.columns for col in INPUT_COLS + OUTPUT_COLS):
        return None, None
    # Clip crystal-size outliers
    for col in CRYSTAL_COLS:
        df[col] = iqr_clip_series_simple(df[col])
    u = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    X, Y = [], []
    # Build sequences: previous lag of y and u to predict next y
    for t in range(lag, len(df)):
        y_seq = y[t-lag:t].flatten()
        u_seq = u[t-lag:t].flatten()
        X.append(np.hstack([y_seq, u_seq]))
        Y.append(y[t])
    return np.array(X), np.array(Y)

# Define NARX ANN architecture
class NARXANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn, dropout_rate):
        super().__init__()
        layers = []
        for h in hidden_layers:
            layers += [nn.Linear(input_size, h), activation_fn(), nn.Dropout(dropout_rate)]
            input_size = h
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# Load batch file list and split by trajectory
product_df  = pd.read_csv(DATA_DIR / prod_csv)
all_files   = list(set(product_df['source_file'].values))
train_val, test_files = train_test_split(all_files, test_size=0.15, random_state=42)
train_files, val_files = train_test_split(train_val, test_size=20/85, random_state=42)
data_folder = DATA_DIR / 'full_dataset'

# ──────────────────────────────────────────────
# Print split counts & percentages
# ──────────────────────────────────────────────
n_total = len(all_files)
n_train = len(train_files)
n_val   = len(val_files)
n_test  = len(test_files)

print(f"Total trajectories: {n_total}")
print(f"Train: {n_train:3d} ({n_train/n_total*100:5.1f}%)")
print(f" Val : {n_val:3d} ({n_val/n_total*100:5.1f}%)")
print(f"Test : {n_test:3d} ({n_test/n_total*100:5.1f}%)")

# Build full dataset arrays by concatenating trajectories
def build_dataset(file_list, lag):
    Xs, Ys = [], []
    for fname in file_list:
        path = str(data_folder / fname)
        if os.path.exists(path):
            X, Y = load_and_prepare_narx_data(path, lag)
            if X is not None:
                Xs.append(X)
                Ys.append(Y)
    if not Xs:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

# %%  
# Cell 10: Visualize Raw NARX Dataset

X, Y = build_dataset(train_files, LAG)
print(f"Feature matrix X shape: {X.shape}")
print(f"Target matrix  Y shape: {Y.shape}")

# Plot histograms of each target variable
fig, axes = plt.subplots(len(OUTPUT_COLS), 1, figsize=(8, 2*len(OUTPUT_COLS)))
for i, col in enumerate(OUTPUT_COLS):
    axes[i].hist(Y[:, i], bins=50, alpha=0.7)
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("Count")
axes[-1].set_xlabel("Value")
plt.suptitle("Distributions of All Target Variables", fontsize=16)
plt.tight_layout()
plt.show()

# Plot a few example sequences with their true next-step targets
n_examples = 3
idxs = np.random.choice(len(X), n_examples, replace=False)
time_hist = np.arange(-LAG, 0)
for idx in idxs:
    x_seq = X[idx]
    y_true = Y[idx]
    n_y = LAG * len(OUTPUT_COLS)
    y_hist = x_seq[:n_y].reshape(LAG, len(OUTPUT_COLS))
    u_hist = x_seq[n_y:].reshape(LAG, len(INPUT_COLS))
    fig, (ax_y, ax_u) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # Output history + next true value
    for j, col in enumerate(OUTPUT_COLS):
        ax_y.plot(time_hist, y_hist[:, j], marker='o', label=col)
        ax_y.scatter(0, y_true[j], marker='x', s=100, color='k')
    ax_y.set_title(f"Example #{idx}: Output history + target")
    ax_y.set_ylabel("Output value")
    ax_y.legend(loc='upper left', ncol=2)
    # Input history
    for j, col in enumerate(INPUT_COLS):
        ax_u.plot(time_hist, u_hist[:, j], marker='.', alpha=0.8, label=col)
    ax_u.set_title("Input history")
    ax_u.set_xlabel("Time (steps before t)")
    ax_u.set_ylabel("Input value")
    ax_u.legend(loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()

# %%  
# Cell 11: Scale Data and Prepare DataLoaders

# Build raw train/val/test arrays
X_train, Y_train = build_dataset(train_files, LAG)
X_val,   Y_val   = build_dataset(val_files,   LAG)
X_test,  Y_test  = build_dataset(test_files,  LAG)

# Fit scalers on training data
scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(Y_train)


# Transform each split
X_train_s, Y_train_s = scaler_x.transform(X_train), scaler_y.transform(Y_train)
X_val_s,   Y_val_s   = scaler_x.transform(X_val),   scaler_y.transform(Y_val)
X_test_s,  Y_test_s  = scaler_x.transform(X_test),  scaler_y.transform(Y_test)

# Wrap into PyTorch DataLoaders
train_loader = DataLoader(TensorDataset(
    torch.from_numpy(X_train_s).float(),
    torch.from_numpy(Y_train_s).float()
), batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(TensorDataset(
    torch.from_numpy(X_val_s).float(),
    torch.from_numpy(Y_val_s).float()
), batch_size=BATCH_SIZE, shuffle=False)

test_loader = DataLoader(TensorDataset(
    torch.from_numpy(X_test_s).float(),
    torch.from_numpy(Y_test_s).float()
), batch_size=BATCH_SIZE, shuffle=False)

# Plot histograms of each target variable scaled
fig, axes = plt.subplots(len(OUTPUT_COLS), 1, figsize=(8, 2*len(OUTPUT_COLS)))
for i, col in enumerate(OUTPUT_COLS):
    axes[i].hist(Y_train_s[:, i], bins=50, alpha=0.5, label='train')
    axes[i].hist(Y_val_s[:,   i], bins=50, alpha=0.5, label='val')
    axes[i].hist(Y_test_s[:,  i], bins=50, alpha=0.5, label='test')
    axes[i].set_title(f"Scaled distribution of {col}")
    axes[i].set_ylabel("Count")
    axes[i].legend(loc='upper right')

axes[-1].set_xlabel("Scaled value")
plt.suptitle("Scaled Target Distributions (Train / Val / Test)", fontsize=16)
plt.tight_layout()
plt.show()

# %%  
# Cell 12: Train NARX_ANN with Early Stopping and Save Results

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = NARXANN(X_train_s.shape[1], Y_train_s.shape[1],
                    HIDDEN_LAYERS, ACTIVATION, DROPOUT).to(device)
optimizer = OPTIMIZER_FN(model.parameters(), lr=LEARNING_RATE)
weights   = torch.tensor(WEIGHTS, dtype=torch.float32, device=device)

def loss_fn(pred, target):
    """Weighted MSE loss for multi-output."""
    return torch.mean((pred - target)**2 * weights)

best_val_loss = float('inf')
patience_cnt = 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    # Compute validation loss
    val_loss = sum(
        loss_fn(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
        for xb, yb in val_loader
    ) / len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt = 0
        best_model_state = model.state_dict()
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best model and save artifacts
model.load_state_dict(best_model_state)
model.eval()

SAVE_PATH     = DATA_DIR /f"narxann_{prod_name}.pth"
SCALER_X_PATH = DATA_DIR /f"scaler_x_{prod_name}.save"
SCALER_Y_PATH = DATA_DIR /f"scaler_y_{prod_name}.save"

torch.save(model.state_dict(), SAVE_PATH)
joblib.dump(scaler_x,  SCALER_X_PATH)
joblib.dump(scaler_y, DATA_DIR / SCALER_Y_PATH)

print(f"Model saved to {SAVE_PATH}")
print(f"Input scaler saved to {SCALER_X_PATH}")
print(f"Output scaler saved to {SCALER_Y_PATH}")

# %%
# Cell 13: Evaluate the Trained Model on the Test Set and Plot Predictions

# Disable gradient tracking for evaluation
with torch.no_grad():
    # Compute average weighted MSE across all test batches
    test_loss = sum(
        loss_fn(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
        for xb, yb in test_loader
    ) / len(test_loader.dataset)
print(f"Final Test Loss: {test_loss:.6f}")

# Gather model predictions and true values from the test loader
y_preds_scaled, y_trues_scaled = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        y_preds_scaled.append(model(xb.to(device)).cpu().numpy())
        y_trues_scaled.append(yb.numpy())

# Stack batch-wise arrays into full trajectories
y_preds = np.vstack(y_preds_scaled)
y_trues = np.vstack(y_trues_scaled)

# Verify that shapes match: (num_samples, num_outputs)
print("y_trues.shape =", y_trues.shape)
print("y_preds.shape =", y_preds.shape)
assert y_trues.shape == y_preds.shape, "True and predicted arrays must have the same shape"

# Inverse-transform each output channel back to original scale
y_scalers = {
    i: StandardScaler().fit(Y_train[:, i:i+1])
    for i in range(Y_train.shape[1])
}
for i, sc in y_scalers.items():
    y_preds[:, i] = sc.inverse_transform(y_preds[:, i:i+1]).ravel()
    y_trues[:, i] = sc.inverse_transform(y_trues[:, i:i+1]).ravel()

# Compute and display MSE and MAE for each state variable
print("\nClosed-Loop Evaluation on Full Test Set:")
print(f"{'State':<8}{'MSE':>15}{'MAE':>15}")
for i, name in enumerate(OUTPUT_COLS):
    mse = mean_squared_error(y_trues[:, i], y_preds[:, i])
    mae = mean_absolute_error(y_trues[:, i], y_preds[:, i])
    print(f"{name:<8}{mse:>15.4e}{mae:>15.4e}")

# Plot True vs. Predicted over the entire test trajectory
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.ravel()
t_full = np.arange(len(y_trues))
for i, name in enumerate(OUTPUT_COLS):
    axs[i].scatter(t_full, y_trues[:, i], s=10, color='lightgray', alpha=0.4, label='True')
    axs[i].plot(t_full, y_trues[:, i], color='blue', alpha=0.6)
    axs[i].plot(t_full, y_preds[:, i], color='red', alpha=0.6)
    axs[i].set_title(f"{name}: True vs Predicted")
    axs[i].set_xlabel("Time step")
    axs[i].legend()
plt.suptitle(f"Closed-Loop Results: Full Test Set for {prod_name}", y=1.05)
plt.tight_layout()
plt.show()

# Plot True vs. Predicted for the first 100 samples
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.ravel()
for i, name in enumerate(OUTPUT_COLS):
    axs[i].scatter(range(100), y_trues[:100, i], s=10, color='lightgray', alpha=0.4, label='True')
    axs[i].plot(range(100), y_trues[:100, i], color='blue', alpha=0.6)
    axs[i].plot(range(100), y_preds[:100, i], color='red', alpha=0.6)
    axs[i].set_title(f"{name}: First 100 Samples")
    axs[i].set_xlabel("Sample")
    axs[i].set_ylabel(name)
    axs[i].legend()
plt.suptitle(f"Closed-Loop Results: First 100 Samples for {prod_name}", y=1.05)
plt.tight_layout()
plt.show()

# %%
# Cell 14: Closed-Loop and Open-Loop Predictions for a Single Test Trajectory

from collections import deque
import random

# Select a random test file from the reserved test set
test_file   = random.choice(test_files)
test_folder = data_folder
print(f"▶ Using random test file: {test_file}")

# Load the selected trajectory and clip crystal-size outliers
df = pd.read_csv(os.path.join(test_folder, test_file), sep='\t')
for col in CRYSTAL_COLS:
    df[col] = iqr_clip(df[col])

# Extract the full raw trajectory of true outputs
y_full = df[OUTPUT_COLS].values  # shape: (N, num_outputs)

# Build lagged NARX sequences for this single file
X_seq, Y_seq = load_and_prepare_narx_data(
    os.path.join(test_folder, test_file), LAG
)
# X_seq and Y_seq have shape: (N-LAG, num_outputs)

# Scale the input sequences once using the previously fitted scaler
X_seq_s = scaler_x.transform(X_seq)  # shape remains (N-LAG, input_dim)

# --- One-shot (Closed-Loop) Prediction ---
with torch.no_grad():
    Xt       = torch.tensor(X_seq_s, dtype=torch.float32).to(device)
    Yp_seq_s = model(Xt).cpu().numpy()

# Inverse-scale each predicted output channel
Yp_seq = np.zeros_like(Yp_seq_s)
for i, sc in y_scalers.items():
    Yp_seq[:, i] = sc.inverse_transform(Yp_seq_s[:, i:i+1]).ravel()

# Prepend the first LAG ground-truth values to align with y_full
Yp_full = np.vstack([y_full[:LAG], Yp_seq])  # final shape: (N, num_outputs)

# --- Recursive (Open-Loop) Prediction ---
N      = len(df)
u_seq  = df[INPUT_COLS].values  # raw input sequence
deq_y  = deque(y_full[:LAG], maxlen=LAG)  # seed with first LAG true outputs
Y_ol   = []

for t in range(LAG, N):
    # Construct one-step feature vector from history
    y_hist = np.hstack(deq_y)             # shape: (LAG * num_outputs,)
    u_hist = u_seq[t-LAG:t].flatten()     # shape: (LAG * num_inputs,)
    x_raw  = np.hstack([y_hist, u_hist])[None, :]  # shape: (1, input_dim)
    x_s    = scaler_x.transform(x_raw)             # scaled features

    with torch.no_grad():
        y_s = model(torch.tensor(x_s, dtype=torch.float32).to(device)).cpu().numpy()

    # Inverse-scale the single-step prediction
    y_hat = np.zeros_like(y_s[0])
    for i, sc in y_scalers.items():
        val = sc.inverse_transform(y_s[:, i:i+1])[0,0]
        y_hat[i] = val

    Y_ol.append(y_hat)
    deq_y.append(y_hat)  # feed back for the next prediction

# Stack closed-loop predictions and prepend initial true values
Y_ol      = np.vstack(Y_ol)                   # shape: (N-LAG, num_outputs)
Y_ol_full = np.vstack([y_full[:LAG], Y_ol])   # shape: (N, num_outputs)

# --- Compute and Display Metrics ---
print("\nMetrics for Open‐Loop and Closed‐Loop Predictions Based on a Single Unseen-File {test_file}:")
print(f"\n{'Open-Loop Metrics:':<32}{'  |  '}{'  Closed-Loop Metrics:':<32}")
print(f"{'State':<8}{'MSE':>12}{'MAE':>12}{'  |  '}{'MSE':>12}{'MAE':>12}")
for i, name in enumerate(OUTPUT_COLS):
    mse_open  = mean_squared_error(Y_seq[:, i], Y_ol[:, i])
    mae_open  = mean_absolute_error(Y_seq[:, i], Y_ol[:, i])
    mse_close = mean_squared_error(Y_seq[:, i], Yp_seq[:, i])
    mae_close = mean_absolute_error(Y_seq[:, i], Yp_seq[:, i])
    print(f"{name:<8}{mse_open:12.4e}{mae_open:12.4e}  |  {mse_close:12.4e}{mae_close:12.4e}")

# --- Plot Alignment of Predictions vs True Trajectory ---
t = np.arange(N)
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.ravel()
for i, name in enumerate(OUTPUT_COLS):
    ax = axes[i]
    ax.plot(t, y_full[:, i],    color='blue',  label='True')
    ax.plot(t, Yp_full[:, i],    color='red',   label='Open-Loop')
    ax.plot(t, Y_ol_full[:, i],  color='green', label='Closed-Loop')
    ax.set_title(name)
    ax.legend()
plt.suptitle(f"Open-Loop vs Closed-Loop Predictions for {test_file}", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Cell 15: Full NARX‐ANN + Conformalized Quantile Regression (CQR) Workflow

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ──────────────────────────────────────────────
# 0. Configuration Parameters for QR & CQR
# ──────────────────────────────────────────────
LAG           = 5
HIDDEN_LAYERS = [10]
ACTIVATION    = nn.ReLU
OPTIMIZER_FN  = torch.optim.Adam
LEARNING_RATE = 0.001
BATCH_SIZE    = 64
EPOCHS        = 100
PATIENCE      = 10
DROPOUT       = 0.05

QUANTILES = [0.05, 0.95]    # quantiles for raw QR
ALPHA     = 0.05             # miscoverage rate for 95% CQR

INPUT_COLS  = ['mf_PM','mf_TM','Q_g','w_crystal','c_in','T_PM_in','T_TM_in']
OUTPUT_COLS = ['T_PM','c','d10','d50','d90','T_TM']
CRYSTAL_COLS = ['d10','d50','d90']

PRODUCT_CSV = DATA_DIR /"Product1.csv"
DATA_FOLDER = DATA_DIR / "full_dataset"
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────
# 1. Helper Functions
# ──────────────────────────────────────────────
def iqr_clip(x):
    """Clip array `x` to [Q1 − 1.5·IQR, Q3 + 1.5·IQR]."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return np.clip(x, q1 - 1.5*iqr, q3 + 1.5*iqr)

def load_and_prepare_data(path, lag):
    """
    Read one trajectory from `path`, clip outliers on outputs,
    and build lagged input/output sequences.
    """
    df = pd.read_csv(path, sep='\t')
    if not all(c in df.columns for c in INPUT_COLS + OUTPUT_COLS):
        return None, None
    # Clip outliers on crystal sizes
    for c in CRYSTAL_COLS:
        df[c] = iqr_clip(df[c].values)
    u = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    X, Y = [], []
    for t in range(lag, len(df)):
        X.append(np.hstack([y[t-lag:t].flatten(), u[t-lag:t].flatten()]))
        Y.append(y[t])
    return np.array(X), np.array(Y)

def pinball_loss(pred, target, q):
    """Compute the quantile (pinball) loss at level `q`."""
    e = target - pred
    return torch.max(q*e, (q-1)*e).mean()

class QuantileANN(nn.Module):
    """
    Simple feedforward network for quantile regression.
    """
    def __init__(self, in_dim, out_dim, hidden, act, drop):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), act(), nn.Dropout(drop)]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# ──────────────────────────────────────────────
# 2. Read and Split Trajectories by Run
# ──────────────────────────────────────────────
df = pd.read_csv(PRODUCT_CSV)
all_runs = sorted(df['source_file'].unique())

# 70% train, 15% validation, 7.5% calibration, 7.5% test
train_files, temp       = train_test_split(all_runs, test_size=0.30, random_state=42)
val_files, cal_test     = train_test_split(temp,      test_size=0.50,      random_state=42)
cal_files, test_files   = train_test_split(cal_test, test_size=0.50,      random_state=42)

#  Print split counts and percentages
total = len(all_runs)
splits = {
    'Train':      train_files,
    'Validation': val_files,
    'Calibration': cal_files,
    'Test':       test_files
}

print("Dataset split:") 
for name, subset in splits.items():
    count = len(subset)
    pct   = count / total * 100
    print(f"  {name:12s}: {count:3d} files ({pct:5.2f}%)")
    
# ──────────────────────────────────────────────
# 3. Gather Raw Data into Arrays
# ──────────────────────────────────────────────
def gather(files):
    """Concatenate X, Y arrays for a list of trajectory files."""
    Xs, Ys = [], []
    for f in files:
        path = os.path.join(DATA_FOLDER, f)
        X, Y = load_and_prepare_data(path, LAG)
        if X is not None:
            Xs.append(X); Ys.append(Y)
    if not Xs:
        return np.empty((0,)), np.empty((0,))
    return np.vstack(Xs), np.vstack(Ys)

X_tr, Y_tr = gather(train_files)
X_val, Y_val = gather(val_files)
X_cal, Y_cal = gather(cal_files)
X_te,  Y_te  = gather(test_files)

# ──────────────────────────────────────────────
# 4. Scaling with RobustScaler
# ──────────────────────────────────────────────
x_scaler = RobustScaler().fit(X_tr)
y_scaler = RobustScaler().fit(Y_tr)

X_tr_s = x_scaler.transform(X_tr)
X_val_s= x_scaler.transform(X_val)
X_cal_s= x_scaler.transform(X_cal)
X_te_s = x_scaler.transform(X_te)

Y_tr_s = y_scaler.transform(Y_tr)
Y_val_s= y_scaler.transform(Y_val)
Y_cal_s= y_scaler.transform(Y_cal)
Y_te_s = y_scaler.transform(Y_te)

# ──────────────────────────────────────────────
# 5. Visualize scaled target distributions
# ──────────────────────────────────────────────
fig, axes = plt.subplots(len(OUTPUT_COLS), 1, figsize=(8, 2*len(OUTPUT_COLS)))
for i, col in enumerate(OUTPUT_COLS):
    axes[i].hist(Y_tr_s[:,  i], bins=50, alpha=0.4, label='train')
    axes[i].hist(Y_val_s[:, i], bins=50, alpha=0.4, label='val')
    axes[i].hist(Y_cal_s[:, i], bins=50, alpha=0.4, label='calibration')
    axes[i].hist(Y_te_s[:,  i], bins=50, alpha=0.4, label='test')
    axes[i].set_title(f"Scaled distribution of {col}")
    axes[i].set_ylabel("Count")
    axes[i].legend(loc='upper right')

axes[-1].set_xlabel("Scaled value")
plt.suptitle("Scaled Target Distributions (Train / Val / Cal / Test)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ──────────────────────────────────────────────
# 5. Prepare DataLoaders
# ──────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                  torch.tensor(Y_tr_s, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                  torch.tensor(Y_val_s, dtype=torch.float32)),
    batch_size=BATCH_SIZE
)

# ──────────────────────────────────────────────
# 6. Train Two Quantile Models (for each q)
# ──────────────────────────────────────────────
qr_models = {}
for q in QUANTILES:
    print(f"\n--- Training QR model q={q} ---")
    m = QuantileANN(X_tr_s.shape[1], Y_tr_s.shape[1],
                    HIDDEN_LAYERS, ACTIVATION, DROPOUT).to(device)
    opt = OPTIMIZER_FN(m.parameters(), lr=LEARNING_RATE)
    best_loss, patience = float('inf'), 0

    for ep in range(1, EPOCHS+1):
        m.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = pinball_loss(m(xb), yb, q)
            loss.backward()
            opt.step()
        # Validation
        m.eval()
        with torch.no_grad():
            losses = [
                pinball_loss(m(xb.to(device)), yb.to(device), q).item() * xb.size(0)
                for xb, yb in val_loader
            ]
        val_loss = sum(losses) / len(val_loader.dataset)
        print(f"QR(q={q}) Ep{ep:03d}: {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss, patience, best_state = val_loss, 0, m.state_dict()
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping QR(q={q}).")
                break

    m.load_state_dict(best_state)
    qr_models[q] = m

# ──────────────────────────────────────────────
# 7. Calibration on Calibration Set
# ──────────────────────────────────────────────
with torch.no_grad():
    Xc_t    = torch.tensor(X_cal_s, dtype=torch.float32).to(device)
    lo_cal_s = qr_models[QUANTILES[0]](Xc_t).cpu().numpy()
    hi_cal_s = qr_models[QUANTILES[1]](Xc_t).cpu().numpy()


lo_cal = y_scaler.inverse_transform(lo_cal_s)
hi_cal = y_scaler.inverse_transform(hi_cal_s)

# Compute conformity errors and per-dimension inflation
E_cal = np.maximum(lo_cal - Y_cal, Y_cal - hi_cal)  # shape (n_cal, num_outputs)
n_cal = E_cal.shape[0]
k     = int(np.ceil((n_cal+1)*(1-ALPHA))) - 1

Q_hat = np.zeros(len(OUTPUT_COLS))
for j in range(len(OUTPUT_COLS)):
    errors_sorted = np.sort(E_cal[:, j])
    Q_hat[j] = max(errors_sorted[min(k, n_cal-1)], 0.0)
print("Per-dimension CQR inflation constants:", Q_hat)

# ──────────────────────────────────────────────
# 8. Final CQR Intervals & Visualization (First N=900 samples)
# ──────────────────────────────────────────────
N = 900
with torch.no_grad():
    Xt = torch.tensor(X_te_s[:N], dtype=torch.float32).to(device)
    lo_s = qr_models[QUANTILES[0]](Xt).cpu().numpy()
    hi_s = qr_models[QUANTILES[1]](Xt).cpu().numpy()
    y_preds_s  = model(Xt).cpu().numpy()


# Inverse-scale raw QR
lo_uncal = y_scaler.inverse_transform(lo_s)
hi_uncal = y_scaler.inverse_transform(hi_s)

# Apply conformal calibration
lo_cqr = lo_uncal - Q_hat
hi_cqr = hi_uncal + Q_hat

y_preds   = y_scaler.inverse_transform(y_preds_s)
Y_te_orig = y_scaler.inverse_transform(Y_te_s)

# Plot for each output variable
t = np.arange(N)
for j, name in enumerate(OUTPUT_COLS):
    y_true = Y_te_orig[:N, j]
    y_ann  = y_preds[:N, j]
    lo_q   = lo_uncal[:N, j]; hi_q   = hi_uncal[:N, j]
    lo_c   = lo_cqr[:N, j];  hi_c   = hi_cqr[:N, j]

    cov_raw = ((y_true >= lo_q) & (y_true <= hi_q)).mean() * 100
    cov_cqr = ((y_true >= lo_c) & (y_true <= hi_c)).mean() * 100


    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    # Panel 1: ANN prediction only
    axs[0].plot(t, y_ann, linewidth=1, color='RED', label='ANN Pred.' )
    axs[0].scatter(t, y_true, s=7, alpha=0.4, color='gray', label='Data')
    axs[0].set_title(f"{name}\nANN Only")
    axs[0].legend()
    # Panel 2: Raw QR intervals
    in_q = (y_true >= lo_q) & (y_true <= hi_q)
    axs[1].fill_between(t, lo_q, hi_q, alpha=0.3, color='blue',label='Raw QR ')
    axs[1].plot(t, y_ann, linewidth=1, color='RED', label='ANN Pred.')
    axs[1].plot(t, lo_q, alpha=0.8, linewidth=0.5, color='blue',label='Lower Bound ANN')
    axs[1].plot(t, hi_q, alpha=0.8, linewidth=0.5, color='blue',label='Upper Bound ANN')
    axs[1].scatter(t[in_q],  y_true[in_q],  s=7, alpha=0.4, color='gray',label='Data (in interval)')
    axs[1].scatter(t[~in_q], y_true[~in_q], s=30, marker='x', color='red',label='Data (out of interval)')
    axs[1].set_title(f"Raw QR Covarege({cov_raw:.1f}%)")
    axs[1].legend(loc='upper right')
    # Panel 3: Conformalized QR intervals
    in_c = (y_true >= lo_c) & (y_true <= hi_c)
    axs[2].fill_between(t, lo_c, hi_c, alpha=0.3, color='green',label=f'CQR ({cov_cqr:.1f}%)')
    axs[2].plot(t, y_ann, linewidth=1, color='RED', label='ANN Pred.')
    axs[2].plot(t, lo_c, alpha=0.8, linewidth=0.5, color='green',label='Lower Bound ANN')
    axs[2].plot(t, hi_c, alpha=0.8, linewidth=0.5, color='green',label='Upper Bound ANN')
    # scatter in-bounds
    axs[2].scatter(t[in_c],  y_true[in_c],  s=7, alpha=0.4, color='gray',label='Data (in interval)')
    # scatter out-of-bounds
    axs[2].scatter(t[~in_c], y_true[~in_c], s=30, marker='x', color='red',label='Data (out of interval)')
    axs[2].set_title(f"CQR ({cov_cqr:.1f}%)")
    axs[2].legend(loc='upper right')

    plt.suptitle(f"CQR Results for {name}", y=1.02)
    plt.tight_layout()
    plt.show()

import joblib
import pickle

# Save all Quantile‐Regression models’ state_dicts

state_dicts = {str(q): m.state_dict() for q, m in qr_models.items()}
torch.save(state_dicts, DATA_DIR / "qr_model.pth")

# 8.2 Save your input/output scalers
joblib.dump(x_scaler,DATA_DIR / "x_scaler.pkl")
joblib.dump(y_scaler,DATA_DIR / "y_scaler.pkl")

print(f"QR Model saved ")
print(f"QR_Input scaler saved")
print(f"QR_Output scaler saved")
# %%
