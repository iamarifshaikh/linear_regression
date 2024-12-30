# Step 1: Data Cleaning and Preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("./dataset/HousingData.csv")

# Inspect dataset
print("Dataset Head:")
print(df.head())

# Step 2: Check for missing value
print("\nMissing values per column: ")
print(df.isnull().sum())

# Fill missing values (for simplicity, use median)
df.fillna(df.median(), inplace=True)
print("\nMissing Values After Imputation: ")
print(df.isnull().sum())

# Step 3: Split features and target
X = df.drop(columns="MEDV") # MEDV is the target column
y = df["MEDV"]

# Step 4: Scale features (Standardization: mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verify the shape
print("\nTraining set shape (X_train): ", X_train.shape)
print("Testing set shape (X_test): ", X_test.shape)

# Step 2: Cost Function

def compute_cost(X, y, weights, bias):
    """
    Compute the cost function J for linear regression.
    Args:
    X: ndarray of shape (m,n) feature matrix
    y: ndarray of shape (m,), target values
    wights: ndarray of shape (n,), weights for features
    bias: float, bias term

    Returns: 
    cost : float, cost function value  
    """
    m = len(y) # Number of samples
    y_pred = np.dot(X, weights) + bias # Linear Prediction
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) # MSE Formula
    return cost

# Initialize dummy values for demonstration
m, n = X_train.shape # Assume data is already preprocessed
initial_weights = np.zeros(n) # Initialize weights to zero
initial_bias = 0 # Initialize bias to zero

# Compute cost using dummy weights
initial_cost = compute_cost(X_train, y_train.values, initial_weights, initial_bias)

print(f"Initial Cost (MSE): {initial_cost}")

# Step 3: Gradient Descent

def gradient_descent(X, y, weights, bias, learning_rate, epochs):
    """
    Perform gradient descent to update weights and bias.
    Args:
    X : ndarray of shape (m, n), feature matrix
    y : ndarray of shape (m,), target values
    weights : ndarray of shape (n,), initial weights
    bias : float, initial bias
    learning_rate : float, step size for gradient descent
    epochs : int, number of iterations

    Returns:
    weights : ndarray of shape (n,), optimized weights
    bias : float, optimized bias
    cost_history : list of float, cost function value at each epoch
    """
    m = len(y)
    cost_history = []  # To store cost values during training

    for epoch in range(epochs):
        # Predictions
        y_pred = np.dot(X, weights) + bias

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Gradient for weights
        db = (1 / m) * np.sum(y_pred - y)        # Gradient for bias

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Compute the current cost
        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)

        # Print cost every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")

    return weights, bias, cost_history

# Initialize hyperparameters
learning_rate = 0.01
epochs = 1000

# Train the model using gradient descent
optimized_weights, optimized_bias, cost_history = gradient_descent(X_train, y_train.values, initial_weights, initial_bias, learning_rate, epochs)

print(f"Final Cost: {cost_history[-1]}")

# Step 4: Linear Regression Prediction

def predict(X, weights, bias):
    """
    Predict target values using the optimized weights and bias.

    Args:
    X : ndarray of shape (m, n), feature matrix
    weights : ndarray of shape (n,), optimized weights
    bias : float, optimized bias

    Returns:
    y_pred : ndarray of shape (m,), predicted values
    """
    return np.dot(X, weights) + bias

# Predict on training and test datasets
y_train_pred = predict(X_train, optimized_weights, optimized_bias)
y_test_pred = predict(X_test, optimized_weights, optimized_bias)

# Step 5: Evaluate the model 

# Evaluate on training data
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Evaluate on test data
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print results
print(f"Training Data: MSE = {mse_train}, R-squared = {r2_train}")
print(f"Test Data: MSE = {mse_test}, R-squared = {r2_test}")

# Step 6: Visualization of the best fit line 

def plot_best_fit_line(X, y_actual, y_pred, feature_name, target_name):
    """
    Plots the actual vs predicted data and the best-fit line.

    Args:
    X : ndarray of shape (m,), input feature values
    y_actual : ndarray of shape (m,), actual target values
    y_pred : ndarray of shape (m,), predicted target values
    feature_name : str, name of the feature being plotted
    target_name : str, name of the target variable
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of actual values
    plt.scatter(X, y_actual, color="blue", label="Actual Data")

    # Best-fit line
    plt.plot(X, y_pred, color="red", label="Best-Fit Line")

    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.title(f"{feature_name} vs {target_name}")
    plt.legend()
    plt.show()

# Define features list
features = df.drop(columns="MEDV").columns.tolist()

# Extract RM (rooms per dwelling) for train and test data
rm_train = X_train[:, features.index("RM")].reshape(-1, 1)
rm_test = X_test[:, features.index("RM")].reshape(-1, 1)

# Generate predictions for RM
y_test_pred_for_plot = predict(rm_test, [optimized_weights[features.index("RM")]], optimized_bias)

# Plot best-fit line for RM vs MEDV
plot_best_fit_line(rm_test.flatten(), y_test, y_test_pred_for_plot.flatten(), "RM", "MEDV")


#Step 7:  Scatter plot and best-fit line visualization function
def visualize_rm_vs_medv(df, weights, bias):
    """
    Visualize the relationship between RM and MEDV with a scatter plot
    and best-fit line.
    Args:
    df : DataFrame, original dataset
    weights : ndarray of shape (n,), optimized weights
    bias : float, optimized bias
    """
    # Extract RM (original feature) and MEDV (target)
    rm_values = df["RM"].values
    medv_values = df["MEDV"].values

    # Predicted MEDV based on RM using the optimized weight and bias
    pred_medv = rm_values * weights[features.index("RM")] + bias

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=rm_values, y=medv_values, color="blue", alpha=0.7, label="Actual Data")
    
    # Plot the best-fit line
    plt.plot(rm_values, pred_medv, color="red", linewidth=2, label="Best-Fit Line")
    
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("Median Home Value (MEDV in $1000)")
    plt.title("Best-Fit Line: RM vs MEDV")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Call the function for RM vs MEDV visualization
visualize_rm_vs_medv(df, optimized_weights, optimized_bias)
