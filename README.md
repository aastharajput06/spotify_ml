# Spotify Streams Prediction

This repository contains the code and report for the "Machine Learning from Scratch" project, part of the Machine Learning course (CSCI_6364_10) at George Washington University. The project predicts the number of streams for songs in the Spotify 2023 dataset using Linear Regression implemented from scratch.

## Project Structure
- `data/`: Contains the dataset (`spotify-2023.csv`) and preprocessed data (`X_preprocessed.npy`, `y_preprocessed.npy`).
- `src/`: Contains the Python scripts:
  - `preprocess.py`: Preprocesses the dataset (handles missing values, outliers, scaling, encoding).
  - `linear_regression.py`: Implements Linear Regression from scratch, trains the model, and evaluates it.
- `docs/`: Contains the report and visualization:
  - `report.md`: Detailed report for Question 1, including approach, challenges, results, and bias-variance analysis.
  - `predicted_vs_actual.png`: Scatter plot of predicted vs. actual streams.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`

Install the required libraries:
```bash
pip install pandas numpy matplotlib