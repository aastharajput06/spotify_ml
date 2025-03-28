import numpy as np
import matplotlib.pyplot as plt
import sys

# Linear Regression implementation from scratch
class LinearRegressionScratch:
    def __init__(self, learning_rate=0.001, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if epoch % 1000 == 0:
                mse = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch}/{self.epochs}, Training MSE: {mse}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train-test split function
def train_test_split(X, y, test_size=0.2, random_state=None):
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
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

# Bootstrap sampling
def bootstrap_sample(X, y, n_samples, sample_size):
    indices = np.random.randint(0, len(X), size=(n_samples, sample_size))
    return [(X[idx], y[idx]) for idx in indices]

# Bias-variance decomposition
def bias_variance_decomposition(X, y, complexities, n_bootstrap=10):
    bias, variance, total_error = [], [], []
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for complexity in complexities:
        print(f"\nEvaluating complexity: {complexity} features")
        X_train_subset = X_train_full[:, :complexity]
        X_test_subset = X_test[:, :complexity]
        predictions = []
        for i in range(n_bootstrap):
            X_train, y_train = bootstrap_sample(X_train_subset, y_train_full, 1, len(X_train_subset))[0]
            model = LinearRegressionScratch(learning_rate=0.001, epochs=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_subset)
            predictions.append(y_pred)
            print(f"Bootstrap sample {i+1}/{n_bootstrap} completed")
        predictions = np.array(predictions)
        avg_pred = np.mean(predictions, axis=0)
        bias.append(np.mean((avg_pred - y_test) ** 2))  # Bias^2
        variance.append(np.mean(np.var(predictions, axis=0)))  # Variance
        total_error.append(bias[-1] + variance[-1])
    return bias, variance, total_error

# Load preprocessed data
X = np.load('data/X_preprocessed.npy')
y = np.load('data/y_preprocessed.npy')

# Define complexities (number of features)
complexities = [5, 10, 20, 30, X.shape[1]]
bias, variance, total_error = bias_variance_decomposition(X, y, complexities)

# Print results *before* plotting
print("\nBias-Variance Decomposition Results:")
for i, complexity in enumerate(complexities):
    print(f"Complexity: {complexity} features")
    print(f"  Bias^2: {bias[i]}")
    print(f"  Variance: {variance[i]}")
    print(f"  Total Error: {total_error[i]}")
sys.stdout.flush()  # Force flush the output

# Plot results (optional, since we already have the plot)
plt.figure(figsize=(8, 6))
plt.plot(complexities, bias, label='Bias^2')
plt.plot(complexities, variance, label='Variance')
plt.plot(complexities, total_error, label='Total Error')
plt.xlabel('Model Complexity (Number of Features)')
plt.ylabel('Error')
plt.title('Bias-Variance Decomposition')
plt.legend()
plt.savefig('docs/bias_variance.png')
plt.close()  # Close the plot to avoid interruption