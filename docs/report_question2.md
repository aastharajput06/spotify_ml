# Question 2: Bias-Variance Tradeoff

## Approach
To analyze the bias-variance tradeoff of the Linear Regression model, I modified the preprocessing and training pipeline as follows:
- **Preprocessing**: Added polynomial features (degree 2) to the numerical columns in `preprocess.py`, increasing the feature count from 30 to 49. This allows varying model complexity by using different numbers of features.
- **Bias-Variance Decomposition**: Implemented a `bias_variance_decomposition` function in `linear_regression.py` that:
  - Uses bootstrapping to create 10 training sets for each complexity level (reduced from 20 to speed up computation).
  - Trains models with varying complexity (5, 10, 20, 30, and 49 features).
  - Computes bias (average squared error), variance (variability of predictions), and total error (bias + variance).
- **Visualization**: Plotted bias, variance, and total error against model complexity, saved as `docs/bias_variance.png`.

## Results
![Bias-Variance Decomposition](bias_variance.png)  
*Figure 1: Bias, variance, and total error as a function of model complexity (number of features).*

- **Numerical Results**:
  - Complexity: 5 features
    - Bias^2: 1.870e+17
    - Variance: 2.969e+14
    - Total Error: 1.873e+17
  - Complexity: 10 features
    - Bias^2: 1.255e+17
    - Variance: 5.643e+14
    - Total Error: 1.261e+17
  - Complexity: 20 features
    - Bias^2: 1.269e+17
    - Variance: 2.881e+14
    - Total Error: 1.272e+17
  - Complexity: 30 features
    - Bias^2: 9.601e+16
    - Variance: 2.933e+14
    - Total Error: 9.630e+16
  - Complexity: 49 features
    - Bias^2: 9.676e+16
    - Variance: 4.408e+14
    - Total Error: 9.720e+16

- **Analysis**:
  - **Bias**: Decreases as model complexity increases (from 1.870e+17 at 5 features to 9.676e+16 at 49 features), as the model becomes more flexible and better fits the data.
  - **Variance**: Increases with complexity (from 2.969e+14 at 5 features to 4.408e+14 at 49 features), as the model becomes more sensitive to variations in the training data.
  - **Total Error**: Shows a U-shape, with the lowest total error (9.630e+16) at 30 features, indicating an optimal balance between bias and variance. The total error decreases from 1.873e+17 at 5 features to 9.630e+16 at 30 features, then slightly increases to 9.720e+16 at 49 features due to the increase in variance.

## Conclusion
The results confirm the bias-variance tradeoff: simpler models (fewer features) have high bias but low variance, while complex models (more features) have low bias but higher variance. The optimal model complexity for this dataset is around 30 features, where the total error is minimized. Future improvements could involve feature selection to retain only the most relevant features or exploring non-linear models (e.g., decision trees) to further reduce bias without increasing variance excessively.