import numpy as np
import matplotlib.pyplot as plt

# Linear Regression implementation from scratch
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.001, epochs=5000):
        """
        Initialize the Linear Regression model.
        
        Parameters:
        - learning_rate: Step size for gradient descent (default: 0.001 to avoid divergence)
        - epochs: Number of iterations for gradient descent (default: 5000 for better convergence)
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights to zeros
        self.bias = 0  # Initialize bias to zero

        # Gradient Descent
        for epoch in range(self.epochs):
            # Prediction: h(x) = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias
            # Error
            error = y_pred - y

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print progress every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch + 1}/{self.epochs}, Training MSE: {mse:.2e}")

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        
        Returns:
        - Predicted values (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias

# Train-test split function
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - test_size: Proportion of the dataset to include in the test split (default: 0.2)
    - random_state: Seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(test_size * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Evaluation metrics
def mse(y_true, y_pred):
    """
    Compute Mean Squared Error.
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    
    Returns:
    - MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Compute R² score.
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    
    Returns:
    - R² score
    """
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

# Load preprocessed data
try:
    X = np.load('data/X_preprocessed.npy')
    y = np.load('data/y_preprocessed.npy')
except ValueError as e:
    print(f"Error loading .npy files: {e}")
    print("Please ensure you have rerun preprocess.py with .astype(float) to convert X and y to numerical arrays.")
    exit(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to confirm
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Initialize and train the model
model = LinearRegressionScratch(learning_rate=0.001, epochs=5000)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute evaluation metrics
train_mse = mse(y_train, y_train_pred)
test_mse = mse(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("\nEvaluation Metrics:")
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("Training R²:", train_r2)
print("Testing R²:", test_r2)
print("\nFirst 5 Test Predictions:", y_test_pred[:5])
print("First 5 Test Actual Values:", y_test[:5])

# Visualize predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Streams")
plt.ylabel("Predicted Streams")
plt.title("Predicted vs Actual Streams (Test Set)")
plt.legend()
plt.savefig('docs/predicted_vs_actual.png')  # Save the plot for the report
plt.show()