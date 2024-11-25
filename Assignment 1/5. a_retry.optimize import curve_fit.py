from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Import CSV file
csv_file_path = r'updated_dataset.csv'
data = pd.read_csv(csv_file_path)

# Filter out rows where X3 is zero
data = data[data['X3'] != 0]

# Prepare the independent and dependent variables
X1 = data['X1'].values  # Convert to numpy array for compatibility
X2 = data['X2'].values
X3 = data['X3'].values
Y = data['Y'].values

# Define a nonlinear model
def nonlinear_model_density(X, a0, m, l):
    X1, X2, X3 = X  # Unpack independent variables
    return a0 * X1**m * (X2 / X3**l)

# Stack independent variables for curve_fit
X = np.vstack((X1, X2, X3))

# Initial guesses for parameters [a0, m, l]
initial_guesses = [8, 3, 5]

# Fit the nonlinear model using curve_fit
params, covariance = curve_fit(nonlinear_model_density, X, Y, p0=initial_guesses)

# Extract optimized parameters
a0, m, l = params

# Print the results
print(f"Optimized Parameters:\n a0: {a0:.4f}, m: {m:.4f}, l: {l:.4f}")

# Calculate fitted values (predictions)
Y_fitted = nonlinear_model_density(X, *params)

# Evaluate the model (R^2 value)
residuals = Y - Y_fitted
ss_residual = np.sum(residuals**2)
ss_total = np.sum((Y - np.mean(Y))**2)
r_squared = 1 - (ss_residual / ss_total)

print(f"R-squared: {r_squared:.4f}")


# Visualize relationships
#sns.pairplot(data, vars=["X1", "X2", "X3", "Y"])
#plt.show()

# Correlation matrix
#corr_matrix = data[["X1", "X2", "X3", "Y"]].corr()
#sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
#plt.show()