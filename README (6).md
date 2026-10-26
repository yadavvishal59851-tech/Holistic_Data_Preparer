# Holistic Data Preparer

A Jupyter Notebook project demonstrating **data preprocessing and feature engineering** on a customer credit risk dataset. It explores multiple approaches to missing values, outliers, categorical encoding, feature creation, scaling, and transformations before machine learning.

## Project files

- `Holistic_Data_Preparer.ipynb` — main notebook.
- `customer_credit_risk_dataset (1).csv` — input customer dataset (required; not included in this notebook).
- `customer_metadata.json` — customer metadata file (required for the JSON-loading example; not included).

## Tools and libraries

- Python and Jupyter Notebook
- NumPy, Pandas
- SciPy
- scikit-learn

## Topics covered

### 1. Loading and profiling data
- Read customer records from CSV and metadata from JSON.
- Inspect data using `info()`, descriptive statistics, and missing-value counts.

### 2. Handling missing values
- Mean imputation for age, annual income, and credit score.
- Median imputation for loan amount.
- Most-frequent imputation for employment type and gender.
- Missing-value flag and random-sample imputation example.
- KNN imputation (`KNNImputer`).
- Iterative/MICE-style imputation (`IterativeImputer`).

### 3. Outlier handling
- Z-score filtering.
- Interquartile range (IQR) filtering.
- Percentile-based filtering.
- Winsorization.

### 4. Feature engineering
- Extract year, month, day, and weekday from `join_date`.
- Label-encode gender.
- Create equal-width and quantile-based income bins.
- Create a credit score flag for scores above 700.
- Cluster transaction counts using K-Means.

### 5. Feature scaling and transformations
- StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, and RobustScaler.
- Log, square-root, and reciprocal transformations of spending ratio.
- Yeo-Johnson power transformation.
- ColumnTransformer combining numeric scaling and one-hot encoding.

## Setup and usage

1. Install Python and Jupyter Notebook.
2. Install the required packages:

   ```bash
   pip install notebook numpy pandas scipy scikit-learn
   ```

3. Place the CSV and JSON input files in a convenient folder.
4. Open `Holistic_Data_Preparer.ipynb` and **update the file paths** in the `pd.read_csv(...)` and `open(...)` calls. The notebook currently uses Windows-specific paths.
5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open the notebook and run its cells in order.

## Dataset fields used

The notebook references fields including `age`, `annual_income`, `credit_score`, `loan_amount`, `employment_type`, `gender`, `join_date`, `transaction_count`, and `spending_ratio`. Your dataset must contain the fields used by each step.

## Notes

- The notebook is an **educational demonstration** of alternative preprocessing techniques, not a finalized end-to-end production pipeline.
- Some methods are shown as separate alternatives; they are not necessarily intended to be applied sequentially to the same columns.
- Several missing-value methods are demonstrated after earlier imputation, so later examples may have no missing values left to fill.
- The missing-income flag is created after earlier imputation; to capture original missingness, create it before filling values.
- The notebook does not currently train or evaluate a credit-risk prediction model.
- When adapting this work for machine learning, split train/test data before fitting imputers, scalers, encoders, or clustering models to avoid data leakage.

## Goal

Build practical familiarity with data cleaning, preprocessing, and feature engineering methods commonly used to prepare tabular data for machine learning.
