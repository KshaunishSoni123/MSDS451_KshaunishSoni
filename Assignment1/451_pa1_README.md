# MSDS 451 Programming Assignment 1: MCD Daily Return Direction

Predict whether McDonald's (MCD) stock closes **up (1)** or **flat/down (0)** versus the prior day, using only information available before the trading day.

Notebook: `451_pa1_kshaunishSoni.ipynb`

## Files

| File | Description |
|---|---|
| `451_pa1_kshaunishSoni.ipynb` | Main notebook |
| `451_pa1_historical_data_MCD.csv` | Raw daily prices from Yahoo Finance (created by Step 2) |
| `mcd-with-computed-features.csv` | Prices plus engineered features (created by Step 3) |
| `Assignment1/` | Course jump-start notebook and reference material |

## Pipeline

| Step | Purpose |
|---|---|
| 1. Import libraries | Load packages |
| 2. Get data | Download MCD daily prices, 2000 to present |
| 3. Feature engineering | Build 15 lagged features and the binary `Target` |
| 4. EDA | Drop null rows from lagging; descriptive statistics |
| 5. Define `X` and `y` | Remove same-day columns; keep unscaled features for modeling |
| 6. Feature selection | Rank all feature subsets by AIC |
| 7. Select model features | Choose 5 features from the AIC ranking and correlations |
| 8. Time-series CV splits | 5 expanding-window folds with a 10-day gap |
| 9. Baseline XGBoost | Untuned reference accuracy |
| 10. Grid search | Tune XGBoost hyperparameters |
| 11. Final evaluation | Fit with the best hyperparameters; ROC, confusion matrix, report |

### Features

All features use prior-day data to avoid leakage.

| Features | Description |
|---|---|
| `CloseLag1-3` | Closing price 1–3 days ago |
| `HMLLag1-3` | High minus low (daily range), lagged |
| `OMCLag1-3` | Open minus close (intraday move), lagged |
| `VolumeLag1-3` | Shares traded, lagged |
| `CloseEMA2/4/8` | Exponential moving averages of `CloseLag1` (half-lives 1, 2, 4) |

**Target:** `1` if the daily log return `log(Close / CloseLag1)` is positive, else `0`.

## How AIC Is Used (Steps 6–7)

**Akaike Information Criterion** compares models by balancing fit against complexity:

```
AIC = 2k - 2 * log-likelihood
```

where `k` is the number of parameters (features + intercept). Lower is better; each added feature must improve the log-likelihood enough to pay its penalty of 2.

**In this notebook:**
1. Features are standardized (Step 5) so logistic regression treats them on a common scale.
2. A logistic regression is fit on **every non-empty subset** of the 15 features (2^15 - 1 = 32,767 models), and each model's AIC is recorded.
3. The 10 lowest-AIC subsets are printed. Feature indices map to the column order printed in Step 5 (e.g. `4` = `HMLLag2`, `6` = `OMCLag1`).
4. Step 7 combines this ranking with a correlation analysis and keeps one feature per family (two for intraday moves) to avoid redundancy. Selected features:

   `OMCLag1`, `HMLLag2`, `OMCLag2`, `VolumeLag2`, `CloseEMA8`

**Notes:**
- AIC is computed in-sample on the full dataset, so it ranks features but doesn't measure predictive accuracy.
- The price-level features (`CloseLag1-3`, `CloseEMA2/4/8`) are >0.999 correlated with each other, so only `CloseEMA8` is kept.

## How XGBoost Is Used (Steps 9 and 11)

**XGBoost** (extreme gradient boosting) builds an ensemble of decision trees in sequence, with each new tree correcting the errors of the trees before it. It's configured as a binary classifier (`objective='binary:logistic'`) that outputs the probability of an up day; probabilities above 0.5 are predicted as up.

- **Step 9 (baseline):** an untuned model with 1,000 trees, scored with time-series cross-validation, as a reference point.
- **Step 11 (final):** a model built with the grid search's best hyperparameters (`**grid_search.best_params_`), fit on the full dataset and evaluated with a ROC curve, confusion matrix and classification report.

Tree models split on thresholds, so they don't need feature scaling; the models use the unscaled features from Step 5.

**Note:** Step 11 scores the model on the same data it was trained on (in-sample), which overstates performance. Use the cross-validated scores from Steps 9–10 as the out-of-sample estimate.

## How GridSearchCV Is Used (Step 10)

`GridSearchCV` fits **every** combination of the listed hyperparameter values and keeps the best one. Unlike `RandomizedSearchCV`, which samples a fixed number of combinations, it's exhaustive and reproducible, but its cost multiplies with each value added.

| Hyperparameter | Values | Role |
|---|---|---|
| `max_depth` | 3, 5, 7 | Tree depth (complexity) |
| `min_child_weight` | 1, 5, 10 | Minimum leaf size (regularization) |
| `subsample` | 0.7, 1.0 | Fraction of rows used per tree |
| `learning_rate` | 0.01, 0.05, 0.1 | Shrinkage per boosting step |
| `n_estimators` | 100, 300, 500 | Number of trees |

162 combinations × 5 folds = **810 model fits**.

- **Cross-validation:** `TimeSeriesSplit(n_splits=5, gap=10)`. Each fold trains on earlier data and tests on the block that follows, so the model never trains on the future. The 10-day gap keeps moving-average features from overlapping across the train/test boundary.
- **Selection metric:** `refit='accuracy'`. The combination with the highest mean cross-validated accuracy is selected. ROC AUC and balanced accuracy are recorded for every combination for reference.
- **Output:** best parameters, best accuracy, and the top 10 combinations. Step 11 reads the best parameters from `grid_search.best_params_`.

## How to Run the Notebook

### 1. Set up the environment

The notebook was developed with Anaconda Python 3.13. Install the required packages:

```bash
pip install numpy pyarrow polars matplotlib seaborn scikit-learn scipy xgboost yfinance "joblib>=1.5"
```

Versions used: scikit-learn 1.6.1, xgboost 3.4.1, polars 2.0.0, yfinance 1.7.0, joblib 1.6.0.

> joblib 1.4.x prints harmless but noisy `ChildProcessError: [Errno 10] No child processes` messages on Python 3.13 after parallel jobs. Upgrading joblib removes them.

### 2. Open the notebook

```bash
cd MSDS451_KshaunishSoni
jupyter lab 451_pa1_kshaunishSoni.ipynb
```

Select the Python 3 kernel for the environment where the packages are installed.

### 3. Run all cells in order

Use **Kernel → Restart Kernel and Run All Cells**. Cells depend on variables from earlier cells, so run them top to bottom.

| Step | Approximate runtime |
|---|---|
| 2. Get data | Seconds; needs an internet connection |
| 6. AIC feature selection | Several minutes (32,767 model fits) |
| 10. Grid search | 1–3 minutes (810 fits, uses all CPU cores) |
| All other steps | Seconds |

To run the notebook from the command line instead:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 451_pa1_kshaunishSoni.ipynb
```

### Tips

- **Change the stock or dates:** edit `symbol`, `start_date` and `end_date` in Step 2. Re-run Steps 6–7 afterwards and update `selectedFeatures` in Step 7 to match the new AIC results.
- **Re-running Step 7** is safe: it always selects columns from the full unscaled matrix `XRaw`.
- **Step 11 requires Step 10:** the final model reads `grid_search.best_params_`.
- **If the file is edited outside Jupyter** (for example, by a script), use **File → Reload Notebook from Disk** before saving. Otherwise Jupyter overwrites the changes with its in-memory copy.

## Limitations

- Every feature's correlation with daily returns is below 0.035, so the predictive signal is weak. Cross-validated accuracy close to the majority-class rate (~52% up days) is expected for daily direction on a large-cap stock.
- Price, range and volume features are in dollar and share units, which drift over 26 years as MCD's price rose from about $20 to $300+.
- Step 11's results are in-sample and should not be reported as expected performance.


## AI Use

- AI was predominantly used for formatting, markdown comments, and initialization of the jupyter notebook. 
- All code was either written by Kshaunish Soni or generalized from the class provided jump start notebook. 