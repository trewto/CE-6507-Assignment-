from scipy.optimize import curve_fit

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Import CSV file
csv_file_path = r'2. combined_dataset.csv'
data = pd.read_csv(csv_file_path)
data = data[data['Space_Headway'] != 0]

# Prepare the independent and dependent variables
X_headway = data['Space_Headway']


X_density = 1 / X_headway
Y_speed = data['v_Vel']




#
print(X_density)


# Define a nonlinear model (e.g., power law: Speed = a * Density^b)
def nonlinear_model_density(X, a, b):
    #return a * np.log(X / b)
    return a * np.log(b/X)
    return  a * ( 1 - X / b )
    return a * (X**b)


# Fit the nonlinear model using curve fitting
params, covariance = curve_fit(nonlinear_model_density, X_density, Y_speed, p0=[2,5])

# Extract the optimized parameters
a, b = params

# Print the results
print(f"Nonlinear Model: a =  {a} ,b =  {b}")

# Calculate the fitted values (predictions)
Y_fitted_density_nl = nonlinear_model_density(X_density, *params)

# Evaluate the model (R^2 value)
residuals = Y_speed - Y_fitted_density_nl
ss_residual = np.sum(residuals**2)
ss_total = np.sum((Y_speed - np.mean(Y_speed))**2)
r_squared_nl = 1 - (ss_residual / ss_total)

print(f"R-squared: {r_squared_nl}")


# Convert X_density and Y_fitted_density_nl to numpy arrays
X_density_array = np.array(X_density)
Y_fitted_density_nl_array = np.array(Y_fitted_density_nl)

# Sort the data by X_density
sorted_indices = np.argsort(X_density_array)
X_density_sorted = X_density_array[sorted_indices]
Y_fitted_density_nl_sorted = Y_fitted_density_nl_array[sorted_indices]


# Plot the data and the fitted curve
plt.scatter(X_density, Y_speed, label="Original Data", color="blue")
#plt.plot(X_density, Y_fitted_density_nl, label="Fitted Curve", color="red")
plt.plot(X_density_sorted, Y_fitted_density_nl_sorted, label="Fitted Curve", color="red")
plt.xlabel("Density")
plt.ylabel("Speed")
plt.title("Speed vs. Density (Nonlinear)")
plt.legend()
plt.show()
