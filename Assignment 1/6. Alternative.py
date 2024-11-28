from scipy.optimize import least_squares, curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import CSV file
csv_file_path = r'updated_dataset__for_python.csv'
data = pd.read_csv(csv_file_path)

# Filter out rows where X3 or Y is zero
data = data[(data['X3'] != 0) & (data['Y'] != 0)]

# Prepare the independent and dependent variables
X1 = data['X1'].values
X2 = data['X2'].values
X3 = data['X3'].values
Y = data['Y'].values

# Normalize the independent variables for better scaling
X1_norm = X1 / np.max(X1)
X2_norm = X2 / np.max(X2)
X3_norm = X3 / np.max(X3)
Y_norm = Y / np.max(Y)

# Stack normalized independent variables
X = np.vstack((X1_norm, X2_norm, X3_norm))

# Define the residual function for least squares
def residuals(params, X, Y):
    a0, m, l = params
    X1, X2, X3 = X
    Y_pred = a0 * (X1**m) * (X2 / X3**l)
    return Y - Y_pred

# Initial guesses
initial_guesses = [0.6, 1, 1.3]

# Solve using least_squares
result = least_squares(
    residuals,
    initial_guesses,
    args=(X, Y_norm),  # Use normalized data
    method='trf',  # Trust Region Reflective
    bounds=([0, 0, 0], [10, 10, 10])  # Example bounds
)

# Extract optimized parameters
a0, m, l = result.x

# Print the results
print("Optimized Parameters:")
print(f"a0: {a0:.4f}, m: {m:.4f}, l: {l:.4f}")
print(f"Residual Sum of Squares: {result.cost:.4f}")

# Calculate fitted values (denormalize predictions for comparison)
Y_fitted_norm = a0 * (X1_norm**m) * (X2_norm / X3_norm**l)
Y_fitted = Y_fitted_norm * np.max(Y)  # Denormalize

# Evaluate the model (R^2 value)
ss_residual = np.sum((Y - Y_fitted)**2)
ss_total = np.sum((Y - np.mean(Y))**2)
r_squared = 1 - (ss_residual / ss_total)

print(f"R-squared: {r_squared:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y, Y_fitted, alpha=0.7, label='Fitted vs. Actual', color='b')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], '--r', label='Ideal Fit')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('Model Fit: Actual vs. Predicted')
plt.legend()
plt.grid()
plt.show()

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(Y, Y - Y_fitted, alpha=0.7, color='g')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Actual Y')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid()
plt.show()