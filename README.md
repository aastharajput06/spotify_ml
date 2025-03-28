# Spotify Streams Prediction

This repository contains the code and reports for Assignment 1 of the Machine Learning course (CSCI_6364_10) at George Washington University. The project predicts the number of streams for songs in the "Most Streamed Spotify Songs 2023" dataset using Linear Regression implemented from scratch and analyzes the bias-variance tradeoff.

## Project Structure
- **data/**: Contains the dataset and preprocessed data.
  - `spotify-2023.csv`: Original dataset from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023).
  - `top-spotify-songs-2023.zip`: Zipped version of the dataset.
  - `X_preprocessed.npy`: Preprocessed feature matrix (953 rows, 49 features).
  - `y_preprocessed.npy`: Preprocessed target vector (streams).
- **docs/**: Contains the reports and visualizations.
  - `report.question1.md`: Report for Question 1 (Machine Learning from Scratch).
  - `report_question2.md`: Report for Question 2 (Bias-Variance Tradeoff, practical and theoretical).
  - `predicted_vs_actual.png`: Scatter plot of predicted vs. actual streams (Question 1).
  - `bias_variance.png`: Bias-variance decomposition plot (Question 2, practical).
  - `underfitting_plot.png`: Hypothetical underfitting plot (Question 2E, theoretical).
- **src/**: Contains the Python scripts.
  - `explore_data.py`: Script to explore the dataset (not required for the assignment).
  - `preprocess.py`: Preprocesses the dataset (handles missing values, outliers, scaling, encoding).
  - `linear_regression.py`: Implements Linear Regression from scratch, trains the model, evaluates it (Question 1), and performs bias-variance decomposition (Question 2).
  - `plot_underfitting.py`: Generates the hypothetical underfitting plot for Question 2E.
- **README.md**: This file.

## Requirements
- Python 3.8 or higher
- Libraries:
  - `pandas` (for data loading and preprocessing)
  - `numpy` (for numerical computations)
  - `matplotlib` (for plotting)

Install the required libraries:
```bash
pip install pandas numpy matplotlib
