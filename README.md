# CrysID — Data-based Modeling of Slug Flow Crystallization (MLME25, TU Dortmund)

This project analyzes slug-flow crystallization data, clusters trajectories by product type, and trains NARX-ANN models to predict process dynamics
using Closed-loop and Open-loop predictions, and implements Conformalized Quantile Regression (CQR) to produce reliable prediction intervals.

##  Tech Stack
- Python (>=3.9), PyTorch, NumPy, Pandas, scikit-learn, Matplotlib

##  Repository Structure
```text
.github/workflows/   # CI (lint/test) placeholder
src/crysid/          # Source code (modules/scripts)
scripts/             # Utility scripts (training, evaluation)
data/                # Sample data + model artefacts 
```

##  Quickstart
```bash
# 1) Clone
git clone https://github.com/YOUR_USERNAME/CrysID_MLME25_Gr03.git
cd CrysID_MLME25_Gr03

# 2) (Recommended) Create env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install
pip install -r requirements.txt


```


##  Packaging
This repository uses a `src/` layout, so modules are imported as:
```python
import crysid
```


## Running the Analysis

You can run the entire pipeline in one go:

```bash
python src/crysid/pipeline.py
```

This will perform:

1. **Data Loading & Inspection**  
2. **Time-Series Visualization**  
3. **Outlier Removal & Rolling-Median Smoothing**  
4. **Feature Extraction & PCA Comparisons**  
5. **HDBSCAN Clustering**  
6. **IsolationForest + KMeans (optional)**  
7. **Export of Clustered Batches**  
8. **Classification Model Building, Training & Saving **
9. **NARX-ANN Training & Saving**  
10. **Closed-Loop Test Evaluation & Plots**  
11. **Single-Trajectory Open-/Closed-Loop Comparison**  
12. **Conformalized Quantile Regression Workflow & Plots**

All interim plots will pop up interactively (or inline if you run in a Jupyter notebook).



## Customizing

- **Switch Product**  
  In the NARX section, change `RUN_ID = 1` to `2` to train on Product 2 data.  
- **Hyperparameters**  
  Edit the `product_configs` dictionary (for NARX) or the `LAG`, `HIDDEN_LAYERS`, etc., at the top of the CQR section to tune model structure, learning rates, batch sizes, etc.



## Outputs

After successful execution, all outputs are saved into the /data/ directory. You will have:

- **Cluster CSVs**: `Product1.csv`, `Product2.csv`, `Corrupted_batches.csv`  
- **Trained Models**:  
  - `narxann_product{Product_ID}.pth`  
  - `qr_model.pth`  
- **Scalers**:  
  - `scaler_x_product{Product_ID}.save` / `scaler_y_product{RUN_ID}.save`  
  - `x_scaler.pkl` / `y_scaler.pkl`  
- **Performance Plots**: Displayed on-screen (you may save them manually via the GUI)



## Troubleshooting

- **Missing Packages**:  
  If you see `ModuleNotFoundError`, double-check your virtual environment and re-run the `pip install` line above.
- **Data Not Found**:  
  Ensure `full_dataset/` is in the directory /data and contains all `.txt` files.
- **CUDA Errors**:  
  If you don’t have a GPU, PyTorch will default to CPU. For CUDA support, install a matching `torch` build from https://pytorch.org.


##  License
MIT — see `LICENSE`.


##  Testing / Evaluation
- Evaluation harness and instructions live in [`scripts/testing/`](scripts/testing/).
