# Question 2: Bias-Variance Tradeoff (Theoretical Analysis)

## A. Optimal Balance Between Bias and Variance
Assuming a typical bias-variance tradeoff graph (training and validation scores vs. dataset size):
- **Model 1 (Simple Model, High Bias)**: This model likely achieves the optimal balance at a larger dataset size, around 800-1000 data points. A simple model (e.g., linear regression with few features) has high bias and low variance. As the dataset size increases, the bias remains relatively constant, but the model benefits from more data to reduce variance slightly, stabilizing the validation score. The optimal point is where the validation score stops improving significantly, indicating a balance between bias (which remains high) and variance (which decreases with more data).
- **Model 2 (Complex Model, High Variance)**: This model likely achieves the optimal balance at a smaller dataset size, around 400-600 data points. A complex model (e.g., a neural network or linear regression with many features) has low bias but high variance. With a small dataset, the variance is high due to overfitting, but as the dataset size increases, the variance decreases, and the validation score improves. The optimal point is where the validation score peaks before variance starts to dominate again (if the model overfits on a very large dataset).

## B. Operating Regime at Different Dataset Sizes
### a. Small Dataset Size (e.g., 250 Data Points)
- **Model 1**: High bias regime. A simple model struggles to capture the complexity of the data with a small dataset, leading to underfitting. The training and validation scores are both low, with a small gap (low variance).
- **Model 2**: High variance regime. A complex model overfits on a small dataset, fitting the training data too closely (high training score) but performing poorly on the validation set (low validation score), resulting in a large gap between training and validation scores.

### b. Large Dataset Size (e.g., 1000+ Data Points)
- **Model 1**: Still in the high bias regime. Even with more data, a simple model cannot reduce its bias significantly, so both training and validation scores remain relatively low, with a small gap (low variance).
- **Model 2**: Optimal or slightly high variance regime. With a large dataset, the complex model’s variance decreases as it generalizes better, bringing the validation score closer to the training score. If the dataset is very large, the model might still overfit slightly, leading to a small high-variance regime.

## C. Modifying Model Complexity
- **High Bias Regime**: If the model is in a high bias regime (e.g., Model 1 at any dataset size), I would increase the model’s complexity to reduce bias. This could involve:
  - Adding more features (e.g., polynomial features, as done in my implementation).
  - Using a more complex model (e.g., switching from linear regression to a decision tree or neural network).
  - Reducing regularization (if applicable) to allow the model to fit the data better.
- **High Variance Regime**: If the model is in a high variance regime (e.g., Model 2 at a small dataset size), I would decrease the model’s complexity to reduce variance. This could involve:
  - Reducing the number of features (e.g., using feature selection to keep only the most relevant features).
  - Using a simpler model (e.g., switching from a neural network to linear regression).
  - Adding regularization (e.g., L2 regularization in linear regression) to penalize large weights and reduce overfitting.
  - Increasing the dataset size (if possible) to help the model generalize better.

## D. Does Adding More Data Improve Performance?
- **Model 1 (High Bias)**: Adding more data will have a limited impact on performance. A high-bias model underfits because it is too simple to capture the data’s complexity, not because of a lack of data. The bias will remain high, and the validation score will improve only slightly as variance decreases marginally. To improve performance significantly, the model’s complexity must be increased.
- **Model 2 (High Variance)**: Adding more data will likely improve performance. A high-variance model overfits on small datasets, but with more data, it can learn more general patterns, reducing variance. The validation score will improve as the model generalizes better, potentially reaching an optimal balance between bias and variance.

## E. Hypothetical Plot for an Underfitting Model
For a hypothetical binary classification task where the model underfits (e.g., a simple linear classifier on a non-linearly separable dataset):
- **Training Score**: The training score will be low across all dataset sizes because the model is too simple to capture the data’s complexity (high bias). It will increase slightly as the dataset size grows (e.g., from 0.6 at 100 data points to 0.65 at 1000 data points) due to a slight reduction in variance.
- **Validation Score**: The validation score will also be low and close to the training score, as the model underfits and does not overfit (low variance). It will follow a similar trend to the training score, increasing slightly with dataset size (e.g., from 0.58 at 100 data points to 0.63 at 1000 data points).
- **Plot Description**:
  - X-axis: Training Instances Size (100 to 1000 data points).
  - Y-axis: Score (0 to 1).
  - Training Score Curve: Starts at 0.6 (100 data points), increases gradually to 0.65 (1000 data points).
  - Validation Score Curve: Starts at 0.58 (100 data points), increases gradually to 0.63 (1000 data points), remaining slightly below the training score with a small, consistent gap (indicating low variance but high bias).
- **Visualization**:
  ![Underfitting Plot](please check the underfitting_plot.png in docs directory)  
  *Figure 1: Training and validation scores for an underfitting model as a function of training instances size.*