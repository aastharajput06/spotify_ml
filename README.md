# Spotify Streams Prediction

This repository contains the code and reports for Assignment 1 of the Machine Learning course (CSCI_6364_10) at George Washington University. The project predicts the number of streams for songs in the "Most Streamed Spotify Songs 2023" dataset using Linear Regression implemented from scratch and analyzes the bias-variance tradeoff.

## Project Structure
- **data/**: Contains the dataset and preprocessed data.
  - `spotify-2023.csv`: Original dataset from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023).
  - `top-spotify-songs-2023.zip`: Zipped version of the dataset.
  - `X_preprocessed.npy`: Preprocessed feature matrix (953 rows, 49 features).
  - `y_preprocessed.npy`: Preprocessed target vector (streams).
- **docs/**: Contains the reports and visualizations.
  - `report_question1.md`: Report for Question 1 (Machine Learning from Scratch).
  - `report_question2.md`: Report for Question 2 (Bias-Variance Tradeoff, practical and theoretical).
  - `predicted_vs_actual.png`: Scatter plot of predicted vs. actual streams (Question 1).
  - `bias_variance.png`: Bias-variance decomposition plot (Question 2, practical).
  - `underfitting_plot.png`: Hypothetical underfitting plot (Question 2E, theoretical).
- **src/**: Contains the Python scripts.
  - `explore_data.py`: Script to explore the dataset (not required for the assignment).
  - `preprocess.py`: Preprocesses the dataset (handles missing values, outliers, scaling, encoding).
  - `linear_regression.py`: Implements Linear Regression from scratch, trains the model, evaluates it (Question 1), and performs bias-variance decomposition (Question 2, practical).
  - `plot_underfitting.py`: Generates the hypothetical underfitting plot for Question 2E (theoretical).
- **README.md**: This file.

## Requirements
- Python 3.8 or higher
- Libraries:
  - `pandas==2.2.1` (for data loading and preprocessing)
  - `numpy==2.2.4` (for numerical computations)
  - `matplotlib==3.10.1` (for plotting)

Install the required libraries with the specified versions:
```bash
pip install pandas==2.2.1 numpy==2.2.4 matplotlib==3.10.1
```

## Steps to Run the Project
### 1. Clone the Repository
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/aastharajput06/spotify_ml.git
cd spotify_ml
```

### 2. Question 1: Machine Learning from Scratch
#### Step 1: Preprocess the Data
- **Script**: `src/preprocess.py`
- **Purpose**: Loads `data/spotify-2023.csv`, handles missing values (imputes with median/mode), caps outliers using 1.5*IQR, adds polynomial features (degree 2), one-hot encodes categorical variables (key, mode), and applies min-max scaling.
- **Command**:
  ```bash
  python src/preprocess.py
  ```
- **Expected Output**:
  Generates `data/X_preprocessed.npy` (953 rows, 49 features) and `data/y_preprocessed.npy` (target: streams).
  Prints preprocessing steps to the console, e.g.:
  ```text
  Loading dataset...
  Handling missing values...
  Capping outliers...
  Adding polynomial features...
  One-hot encoding categorical variables...
  Scaling features...
  Saving preprocessed data to data/X_preprocessed.npy and data/y_preprocessed.npy
  ```

#### Step 2: Train and Evaluate the Model
- **Script**: `src/linear_regression.py`
- **Purpose**: Implements Linear Regression from scratch, splits the data into 80% training and 20% testing sets, trains the model on the training set, evaluates it on the test set using MSE and R², and generates a predicted vs. actual plot.
- **Command**:
  ```bash
  python src/linear_regression.py
  ```
- **Expected Output (Question 1 Part)**:
  The script first runs the Question 1 tasks and prints:
  ```text
  Question 1: Training and Evaluating Linear Regression Model
  Epoch 0/2000, Training MSE: 6.051190137787231e+17
  Epoch 1000/2000, Training MSE: 1.3694837752590107e+17
  Test MSE: 9.634022068056562e+16
  Test R²: 0.6462522104076616
  ```
  Generates `docs/predicted_vs_actual.png`.

### 3. Question 2: Bias-Variance Tradeoff
#### Part 1: Practical Implementation (Bias-Variance Decomposition)
- **Script**: `src/linear_regression.py`
- **Purpose**: Performs bias-variance decomposition by training models with varying complexities (5, 10, 20, 30, 49 features) using bootstrapping (10 samples per complexity), computes bias, variance, and total error, and generates a bias-variance plot.
- **Command**:
  ```bash
  python src/linear_regression.py
  ```
- **Expected Output (Question 2 Practical Part)**:
  After the Question 1 output, the script continues with Question 2 and prints:
  ```text
  Question 2: Bias-Variance Decomposition
  Evaluating complexity: 5 features
  ...
  Bias-Variance Decomposition Results:
  Complexity: 5 features
    Bias^2: 1.8701309738932387e+17
    Variance: 296917373899460.6
    Total Error: 1.8731001476322333e+17
  Complexity: 10 features
    Bias^2: 1.2554583570760342e+17
    Variance: 564349147093624.2
    Total Error: 1.2611018485469706e+17
  Complexity: 20 features
    Bias^2: 1.2694114994259062e+17
    Variance: 288091946804847.5
    Total Error: 1.2722924188939547e+17
  Complexity: 30 features
    Bias^2: 9.600556449471256e+16
    Variance: 293275666642647.6
    Total Error: 9.62988401613552e+16
  Complexity: 49 features
    Bias^2: 9.675711855097395e+16
    Variance: 440786977187538.4
    Total Error: 9.719790552816149e+16
  ```
  Generates `docs/bias_variance.png`.

#### Part 2: Theoretical Analysis (Question 2E: Underfitting Plot)
- **Script**: `src/plot_underfitting.py`
- **Purpose**: Generates a hypothetical plot for an underfitting model (training and validation scores vs. dataset size) for a binary classification task, as required by Question 2E.
- **Command**:
  ```bash
  python src/plot_underfitting.py
  ```
- **Expected Output**:
  Generates `docs/underfitting_plot.png`. No console output (the script only generates the plot).

## Reports
- **Question 1**: `docs/report_question1.md`
- **Question 2**: `docs/report_question2.md`
