# Regression on Auto-Generated Data

Homework for a Machine Learning course: generate synthetic regression datasets with different properties, then fit a **linear least-squares regressor by hand** to each one using **10-fold cross validation**, and report its error with several metrics.

No machine-learning library is used. The regressor is computed directly from the normal equation, using NumPy for the matrix math and Matplotlib for the plots.

## What the notebook does

All the code is in [`regressor-finder.ipynb`](regressor-finder.ipynb). It has three steps, each with a markdown cell explaining it:

1. **Generate each dataset's metadata and actual relation**
   - **dimension:** number of features, 1 to 9
   - **noise intensity:** 0 to 1
   - **relation type:** linear or non-linear, picked using adjustable probabilities
   - **actual relation:** the function the regressor will try to estimate
     - linear: `y = w0 + w1·x1 + … + wd·xd`
     - non-linear, in the generalized linear form `y = w0 + Σ wj·φj(xj)`: each feature gets its own basis function `φj`, picked from polynomial (degree 2–10), sin, cos, log, sqrt, abs, tanh, sigmoid and gaussian
     - the weights are drawn uniformly from [-10, 10]
2. **Generate the samples**
   - **samples size:** 100 to 4999
   - **inputs:** uniform in [-1000, 1000]
   - **outputs:** the actual relation applied to the inputs, plus Gaussian noise `ε ~ N(0, σ²)` with `σ = noise_intensity × std(y_clean)`, so the noise intensity means the same thing at any output scale
3. **10-fold cross validation**
   - **folds:** for each of the 10 folds, the regressor is trained on the other 9 folds
   - **training:** with the normal equation `w = (XᵀX)⁻¹Xᵀy`, after a column of ones is added to `X` for the bias
   - **testing:** on the held-out fold, with **RMSE**, **MAE**, **MAPE** and **R²** (coefficient of determination)
   - **averaging:** each metric is averaged over the 10 folds

The notebook prints each dataset's metadata, its first samples, and each fold's errors together with the estimated regressor. Only 1-dimensional datasets are also plotted, showing the samples and the 10 regression lines.

## Setup

Requires Python 3 (developed with 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the notebook you also need a Jupyter environment: Jupyter itself (`pip install notebook`), or VS Code with the Jupyter extension, which installs `ipykernel` into the selected environment.

## Usage

Open `regressor-finder.ipynb`, select the `.venv` kernel, and run the cells in order.

### Configuration

| Setting | Cell | Meaning |
|---|---|---|
| `total_datasets_to_be_generated` | 1 | number of datasets to generate |
| `relation_probabilities` | 1 | chances of picking `linear` / `non-linear` |
| `dimension` | 1 | number of features (random 1–9 by default) |
| `noise_intensity` | 1 | noise intensity (random 0–1 by default) |
| `samples_size` | 2 | number of samples (random 100–4999 by default) |
| `k` | 3 | number of cross-validation folds (10) |

For `dimension`, `noise_intensity` and `samples_size`, the random line is followed by a commented-out constant line, e.g.

```python
samples_size = int(4900 * rng.random(size=None) + 100)
# samples_size = 1000
```

To fix a value for an experiment, comment out the random line and uncomment the constant one.

## Findings

These are the experiments run with the notebook for the homework report:

- **No noise:** with a linear relation and zero noise, all 10 regressors recover the actual relation exactly (R² = 1) with 4000, 400 or only 40 samples. Without noise, `d + 1` distinct samples are enough to determine the weights.
- **More noise, more error:** with fixed dimension and samples size, R² drops steadily as the noise intensity grows (0.3 → 0.6 → 0.9 → 2), close to the theoretical `R² ≈ 1 / (1 + α²)`. RMSE and MAE grow too, but they depend on the scale of each random relation.
- **More samples, more overlap:** with few samples, the 10 cross-validation regressors differ noticeably. As the samples size grows (16 → 200 → 2000), they converge and overlap.
- **Structural error:** for a non-linear relation (`y = -2.63 + 1.71·cos(x1)`), the error of the linear regressor stays at the theoretical structural error (RMSE ≈ 1.21) from 100 up to 1,000,000 samples. More data removes the approximation error, but not the error caused by fitting a linear model to a non-linear relation.
