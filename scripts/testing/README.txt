Testing

A self-contained test harness for classifying and evaluating slug-flow crystallization runs against two pretrained
 NARX-ANN models (“Product 1” and “Product 2”). Given a single `.txt` trajectory, it:

1. Extracts summary-statistics features
2. Classifies the run using a Random Forest (`product_classifier.pkl`)
3. Loads the matching NARX-ANN (`narxann_product{1|2}.pth`) and its scalers
4. Runs both closed-loop and open-loop predictions
5. Reports per-state MSE/MAE and overlays true vs. predicted plots

---

Prerequisites

- Python 3.8+
- Packages (install with `pip install`):
  - numpy
  - pandas
  - scikit-learn
  - torch
  - joblib
  - matplotlib

---

Directory Layout

Testing/
├── main.py                       # Main evaluation script
├── preprocessing.py              # Data_Preprocessor module
├── product_classifier.pkl        # RF classifier
├── narxann_product1.pth          # NARX-ANN weights for Product 1
├── narxann_product2.pth          # NARX-ANN weights for Product 2
├── scaler_x_product1.save        # Input scaler for Product 1
├── scaler_y_product1.save        # Output scaler for Product 1
├── scaler_x_product2.save        # Input scaler for Product 2
├── scaler_y_product2.save        # Output scaler for Product 2
└── README.md                     # You are here

---

Configuration

1. Select your test file  
   At the top of `main.py`, set:
   TEST_FILE = 'path/to/your_test_run.txt'
2. Ensure all model & scaler files live in the same directory as `main.py`.

---

Usage:

-Please add your test file to: `Beat-The-Felix/`
-Change the test-file name where pointed at line '67' in Main.py 

From within `Testing/`:

```bash
python main.py
```

The script will:
1. Preprocess TEST_FILE via preprocessing.py  
2. Classify it as Product 1 or 2 (or exit if corrupted)  
3. Run closed-loop and open-loop predictions  
4. Print per-state MSE & MAE  
5. Display comparison plots

---

Outputs

- Console:  
  ```
  Classified as Product 2
  State | MSE_open … MAE_open … MSE_closed … MAE_closed …
  T_PM  | …
  …
  ```
- Plots:  
  Six subplots showing true vs. closed- vs. open-loop trajectories.

---

Troubleshooting

- Missing Columns:  
  If you see “Required columns missing,” verify that your `.txt` file has all input/output headers:
  `mf_PM, mf_TM, Q_g, w_crystal, c_in, T_PM_in, T_TM_in, T_PM, c, d10, d50, d90, T_TM`
- ModuleNotFoundError:  
  Ensure your current directory is `Beat-The-Felix/` so `preprocessing.py` can be imported.
- CUDA Errors:  
  If you lack a GPU, PyTorch will auto-fall back to CPU.

---

## License & Contact

This code was developed for the TU Dortmund “Machine Learning Methods for Engineers” SS 25 project by Group 3. 
Authors:	- Mouhamad Mostafa Choman			(mostafa.choman@tu-dortmund.de)
			- Hassan Mroueh						(hassan.mroueh@tu-dortmund.de)
			- Wishal Shantanu Nandy				(wishalshantanu.nandy@tu-dortmund.de)
			- Eflin Türkmen						(eflin.tuerkmen@tu-dortmund.de)

Supervised By: Felix Brabender (felix.brabender@tu-dortmund.de).
 
