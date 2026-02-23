# PayJoy Prediction Challenge - Project Blueprint

## Project Context

You are an expert machine learning engineer assisting with a Kaggle prediction challenge for a Stanford machine learning class (OIT367).

- **Objective:** Predict `FPD_15` (First Payment Default within 15 days) for PayJoy phone financing orders.

- **Evaluation Metric:** AUC-ROC.

- **Datasets:** `Orders.csv` (main features & target), `Payment_History.csv` (time-series payment data to be aggregated), `Test_OrderIDs.csv` (target predictions).

## File Architecture

We are utilizing a strict two-file architecture to separate experimentation from production:

1. `model_experiments.ipynb`: For rapid testing, hyperparameter tuning, and comparing models.

2. `best_model_pipeline.ipynb`: A clean, streamlined pipeline that houses _only_ the winning model architecture to generate the final `submission.csv`.

## Current Task: Build `model_experiments.ipynb`

Write the code for `model_experiments.ipynb` using a highly modular, functional approach. Generate the following discrete blocks of code in order. **Do not merge them into a single massive function.**

### Module 1: Data Preparation & Feature Selection

Write a function `prep_data(orders_df, payments_df, selected_features, scale_data=True)` that:

- Takes a flexible list of `selected_features` to easily toggle variables in and out.

- Handles missing values appropriately (e.g., median for numerical, mode/constant for categorical).

- Encodes categorical variables (e.g., One-Hot Encoding).

- Scales numerical data using `StandardScaler` (fitting _only_ on the training set to prevent data leakage).

- Returns `X_train`, `X_test`, `y_train`, and the fitted scaler.

### Module 2: XGBoost Generation

Write a function `build_xgb_model(params)` that:

- Takes a dictionary of hyperparameters (e.g., `max_depth`, `learning_rate`, `n_estimators`, `subsample`).

- Initializes and returns an XGBoost classifier configured with those parameters.

### Module 3: Neural Network Generation (PyTorch)

Write a function `build_nn_model(input_dim, hidden_layers, activation='relu')` that:

- Uses PyTorch to dynamically generate a Multi-Layer Perceptron.

- `hidden_layers` should be a list of integers representing the number of neurons in each layer (e.g., `[24, 12, 6]`).

- Applies the specified activation function between layers and a Sigmoid function at the output.

- Returns the PyTorch model class/instance.

### Module 4: Hyperparameter Tuning & Evaluation Engines

Write two testing functions that track performance:

1. `tune_xgb(X_train, y_train, param_grid, cv_folds=5)`: Uses `RandomizedSearchCV` or `GridSearchCV` to test various hyperparameter combinations for XGBoost. Evaluate using AUC-ROC, print the best parameters, and return the best model and a results dataframe.

2. `tune_nn(X_train, y_train, param_grid, epochs, batch_size)`: A custom PyTorch training loop that iterates over a grid of architectures (varying learning rates and hidden layer structures), evaluates validation AUC for each using a train/val split, and returns the best performing model setup.

## Output Constraints & Style Rules

- **Libraries:** Use `pandas`, `numpy`, `xgboost`, `sklearn`, and `torch`.

- **Documentation:** Include concise docstrings for every function explaining inputs and outputs.

- **Execution:** Provide a final execution block at the bottom of the notebook that demonstrates how to call these functions sequentially using a dummy `selected_features` list.

- **Focus:** Keep the code strictly focused on functionality. Do not include excessive EDA or plotting functions (except for an ROC curve plotter).
