![ref1]

**Report - Machine Learning from Scratch: Kaggle Most Streamed Spotify Songs 2023** 

**Aastha Rajput, G23508408** 

**Computer Science, George Washington University** 

**Machine Learning: CSCI\_6364\_10** 

**Professor Armin Mehrabian** 

![ref1]

# Question 1: Machine Learning from Scratch

## Objective
The goal of this task is to predict the number of streams for songs in the "Most Streamed Spotify Songs 2023" dataset using a linear regression model implemented from scratch.

## Approach
### Data Preprocessing
- **Dataset**: Loaded `spotify-2023.csv` using pandas.
- **Missing Values**: Imputed missing values in `in_shazam_charts` with the median and in `key` with the mode.
- **Outliers**: Capped numerical features at 1.5*IQR to handle outliers.
- **Feature Engineering**: Added polynomial features (degree 2) for numerical columns and one-hot encoded categorical variables (`key`, `mode`).
- **Scaling**: Applied min-max scaling to normalize features between 0 and 1.
- **Output**: Saved preprocessed data as `X_preprocessed.npy` (953 rows, 49 features) and `y_preprocessed.npy` (target: `streams`).

### Model Implementation
- **Algorithm**: Implemented a linear regression model from scratch (`LinearRegressionScratch` class).
- **Cost Function**: Mean Squared Error (MSE), defined as `mse = np.mean((y_pred - y) ** 2)`.
- **Optimization**: Used gradient descent with a learning rate of 0.001 and 2000 epochs to minimize the MSE.
- **Features**: Used all 49 features (including polynomial and one-hot encoded features).

### Model Evaluation
- **Train-Test Split**: Split the data into 80% training and 20% testing sets using a custom `train_test_split` function (random_state=42).
- **Training**: Trained the model on the training set, monitoring the MSE (e.g., 2.493e+17 after 1000 epochs for 5 features).
- **Evaluation Metrics**: Implemented MSE and R² score from scratch to evaluate performance on the test set.
- **Visualization**: Generated a predicted vs. actual plot (`docs/predicted_vs_actual.png`) to visualize model performance.

## Challenges Faced
- **Large MSE Values**: The MSE values were very large (e.g., 2.493e+17), likely due to the scale of the target variable (`streams`). Normalizing the target variable could improve this.
- **Feature Engineering**: Deciding which features to include and adding polynomial features increased model complexity, which required careful analysis to avoid overfitting.
- **Convergence**: Tuning the learning rate and number of epochs to ensure gradient descent converged properly.

## Results
- **Performance**: The model achieved an MSE on the test set (exact value depends on your initial run for Question 1, not shown in the latest output). The R² score indicates how well the model explains the variance in the target variable.
- **Visualization**:
  ![Predicted vs. Actual](please see the predicted_vs_actual.png in docs dir)  
  *Figure 1: Predicted vs. actual streams on the test set.*

## Bias-Variance Tradeoff and Model Selection
- **Bias-Variance Decomposition**: Performed in Question 2, the bias-variance decomposition showed that:
  - At 5 features: High bias (1.870e+17), low variance (2.969e+14), total error = 1.873e+17.
  - At 30 features: Optimal balance, with total error minimized at 9.630e+16.
  - At 49 features: Lower bias (9.676e+16) but higher variance (4.408e+14), total error = 9.720e+16.
- **Overfitting/Underfitting**:
  - With 5 features, the model underfits (high bias, unable to capture the complexity of the data).
  - With 49 features, the model shows signs of overfitting (increasing variance, total error rises slightly due to sensitivity to training data variations).
- **Model Selection**: Based on the bias-variance tradeoff, the optimal model complexity is around 30 features, where the total error is minimized. However, the final model used all 49 features, which may lead to slight overfitting. For better generalization, I would recommend retraining the model with 30 features.
- **Generalization**: The model with 30 features balances bias and variance, ensuring better performance on unseen data. The 80-20 train-test split and random sampling help ensure the model generalizes well, but further cross-validation could improve robustness.

## Learnings
- Implementing linear regression from scratch deepened my understanding of gradient descent and the importance of feature scaling.
- The bias-variance tradeoff analysis highlighted the impact of model complexity on performance, guiding better model selection.
- Handling large-scale target variables requires careful preprocessing (e.g., normalizing the target variable).

## Conclusion
The linear regression model successfully predicts song streams, with the best generalization achieved at 30 features. Future improvements could include normalizing the target variable, applying feature selection to reduce complexity, or exploring non-linear models like decision trees to capture more complex patterns in the data.