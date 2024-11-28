import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Load the dataset
csv_file_path = r'updated_dataset.csv'
data = pd.read_csv(csv_file_path)

# Check the minimum value for each column to decide the shift
#min_Y = data['Y'].min()
#min_X1 = data['X1'].min()
#min_X2 = data['X2'].min()
#min_X3 = data['X3'].min()

# Calculate the shift constant
#shift_constant = max(0, -min_Y, -min_X1, -min_X2, -min_X3) + 1  # Add 1 to ensure no zero values
#print("shift constant: ", shift_constant)

# Shift all columns
#data['Y'] += shift_constant
#data['X1'] += shift_constant
#data['X2'] += shift_constant
#data['X3'] += shift_constant

# Now apply the natural log transformation
#data['ln_Y'] = np.log(data['Y'])
#data['ln_X1'] = np.log(data['X1'])
#data['ln_X2'] = np.log(data['X2'])
#data['ln_X3'] = np.log(data['X3'])

# Check for missing values or infinities after the transformation
print(data.isna().sum())  # Count NaNs
print((data == np.inf).sum())  # Count infinities








# Define the nonlinear model
def nonlinear_model(X, a0, m, l):
    X1, X2, X3 = X
    return a0 * (X1**m) * (X2 / (X3**l))

# Prepare independent variables (X1, X2, X3) as a tuple
X = (data['X1'], data['X2'], data['X3'])

# Dependent variable (Y)
Y = data['Y']

# Initial guesses for the parameters (a0, m, l)
initial_guesses = [0.001, 1, 0.001]  # Example: a0 = 1, m = 1, l = 1

# Fit the model using curve_fit
params, covariance = curve_fit(nonlinear_model, X, Y, p0=initial_guesses)

# Extract the optimized parameters
a0, m, l = params

# Print the results
print("Optimized Parameters:")
print(f"a0 (intercept): {a0}")
print(f"m (exponent for X1): {m}")
print(f"l (exponent for X3): {l}")

# Optional: Print the covariance matrix
print("\nCovariance Matrix:")
print(covariance)

# Calculate the fitted values (predictions)
Y_fitted = nonlinear_model(X, *params)

# Calculate R-squared
residuals = Y - Y_fitted
ss_residual = np.sum(residuals**2)
ss_total = np.sum((Y - np.mean(Y))**2)
r_squared = 1 - (ss_residual / ss_total)

# Print R-squared and other statistics
print(f"\nR-squared: {r_squared}")
print(f"Sum of squared residuals: {ss_residual}")
print(f"Sum of squared total: {ss_total}")
