# Regression Models on Boston Housing Dataset

I built and compared multiple regression models on the classic **Boston Housing** dataset to predict **MEDV** (median home value) from 13 neighborhood-level features. This project was done in **Google Colab** and focuses on learning how different regression algorithms behave on the same dataset.

---

## Overview

The goal is simple: **predict housing prices** using machine learning regression.

In the notebook, I:

- Load the Boston Housing dataset from a public URL
- Assign proper column names and verify data quality (shape + missing values)
- Split the data into train/test sets (80/20)
- Train multiple regression models (linear and non-linear)
- Tune hyperparameters using cross-validation
- Compare all models using common regression metrics and select the best one

---

## Dataset

- **Source**: Loaded directly from `https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data`
- **Rows / Columns**: 506 rows, 14 columns (13 features + 1 target)
- **Target**: `MEDV` = median value of owner-occupied homes (in 1000s)
- **Missing values**: None

### Columns (feature glossary)

Each line below is **1 column → meaning**:

- **CRIM**: per-capita crime rate by town  
- **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.  
- **INDUS**: proportion of non-retail business acres per town  
- **CHAS**: Charles River dummy variable (1 if tract bounds river, else 0)  
- **NOX**: nitric oxides concentration (parts per 10 million)  
- **RM**: average number of rooms per dwelling  
- **AGE**: proportion of owner-occupied units built prior to 1940  
- **DIS**: weighted distances to five Boston employment centres  
- **RAD**: index of accessibility to radial highways  
- **TAX**: full-value property-tax rate per 10,000  
- **PTRATIO**: pupil–teacher ratio by town  
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town  
- **LSTAT**: % lower status of the population  
- **MEDV (target)**: median home value (in 1000s)

---

## What I did in the notebook (step-by-step)

### 1) Load + inspect the data

- I loaded the dataset from the URL and assigned the column names.
- I confirmed the dataset size (**(506, 14)**) and checked missing values (none).

### 2) Train/test split

- I separated features `X` and target `y = MEDV`.
- I used an **80/20 split** with `random_state=42` so results are reproducible.

### 3) Train multiple regression models

I trained these models to compare a **linear baseline** vs a **non-linear** model:

- **Linear Regression**: baseline model without regularization.
- **Ridge Regression**: linear regression with L2 regularization (helps when features are correlated).
- **Lasso Regression**: linear regression with L1 regularization (can shrink some coefficients toward 0).
- **Gradient Boosting Regressor**: an ensemble of decision trees that can learn non-linear relationships.

### 4) Hyperparameter tuning (cross-validation)

To keep the comparison fair (and improve performance), I tuned key parameters using `GridSearchCV` with **5-fold CV**:

- **Ridge**: searched `alpha` values to control L2 strength.
- **Lasso**: searched `alpha` values to control L1 strength.
- **Gradient Boosting**: searched over `n_estimators`, `learning_rate`, `max_depth`, `subsample`.

### 5) Evaluation metrics + model selection

For every model, I evaluated on the held-out test set using:

- **MSE** (Mean Squared Error): penalizes large errors more strongly.
- **RMSE** (Root Mean Squared Error): error in the same units as the target.
- **MAE** (Mean Absolute Error): average absolute prediction error.
- **R² (R-square)**: how much variance the model explains (higher is better).

Then I selected the **best model based on the highest R²**.

---

## Results (Test set)

Here are the test results from my notebook:


| Model                 | MSE        | RMSE       | MAE        | R²         |
| --------------------- | ---------- | ---------- | ---------- | ---------- |
| Linear Regression     | 24.2911    | 4.9286     | 3.1891     | 0.6688     |
| Ridge Regression      | 24.3346    | 4.9330     | 3.1829     | 0.6682     |
| Lasso Regression      | 24.2928    | 4.9288     | 3.1886     | 0.6687     |
| **Gradient Boosting** | **6.4138** | **2.5325** | **1.8258** | **0.9125** |


**Best model**: Gradient Boosting (**R² ≈ 0.9125**, i.e., about **91.25%** variance explained)

---

## Why the linear models scored lower than Gradient Boosting

It’s normal in this dataset for simple linear models to score lower (around **R² ≈ 0.67**) compared to a strong non-linear model (here **R² ≈ 0.91**). The main reasons:

- **Non-linear relationships**: Housing prices often change non-linearly with features (for example, room count, pollution, tax rate, and neighborhood indicators don’t affect price in a perfectly straight-line way).
- **Feature interactions**: Some features matter more depending on other features (e.g., the impact of `RM` can depend on `LSTAT` or `DIS`). Linear models don’t naturally capture these interactions unless we explicitly add interaction terms.
- **Threshold effects**: For some neighborhoods, changes in variables may only start affecting price after a certain point; tree-based models handle these “if/then” style splits naturally.
- **Regularization doesn’t fix non-linearity**: Ridge/Lasso help with stability and correlated features, but they’re still linear in the inputs, so they can’t model complex patterns the way boosted trees can.

So the gap here is not a “mistake” in linear regression, it’s a **model capability** difference.

---

## Libraries used

- `pandas`: load and manipulate the dataset in DataFrames.  
- `numpy`: numerical operations (e.g., RMSE via square root).  
- `scikit-learn`: train/test split, pipelines, models, grid search, and evaluation metrics.

In particular, I used:

- `train_test_split`: create reproducible train/test sets.
- `Pipeline` + `StandardScaler`: scale features for Ridge/Lasso consistently.
- `GridSearchCV`: tune hyperparameters with cross-validation.
- `mean_squared_error`, `mean_absolute_error`, `r2_score`: evaluate regression performance.

---

## Project structure

```
Regression Models on Boston Housing Dataset/
├── Regression Models on Boston Housing Dataset.ipynb
└── README.md
```

---

## How to run (Google Colab)

1. Open `Regression Models on Boston Housing Dataset.ipynb` in **Google Colab**
2. Run the notebook cells from top to bottom
3. The notebook will download the dataset automatically from the URL and print a model comparison table

---

## Author

**Md. Tausif Jafar**