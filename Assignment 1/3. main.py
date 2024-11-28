
import numpy as np
import seaborn as sns
import pandas as pd

# Import CSV file
csv_file_path = r'combined_dataset.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe

print(data.head())

# Display all column names
print("Column names in the dataset:")
print(data.columns.tolist())




# Add new columns based on the provided formulas
data['Y'] = data['v_Acc']  # Y = v_Acc (Dependent Variable)

# X1 = v_Vel (Independent Variable 1)
data['X1'] = data['v_Vel']

# X2 = leader_v_Vel - v_Vel (Difference between leader and current vehicle's velocity)
data['X2'] = data['leader_v_Vel'] - data['v_Vel']


# X3 = Euclidean distance between the current vehicle and the leader vehicle
#data['X3'] = np.sqrt((data['Global_X'] - data['leader_Global_X'])**2 + (data['Global_Y'] - data['leader_Global_Y'])**2)
data['X3'] = data['Space_Headway']
# Display the updated dataframe
print("Updated Dataset with new columns:")
print(data[['Y', 'X1', 'X2', 'X3']].head())

data = data.head(500)
data = data[['Y', 'X1', 'X2', 'X3']]


# Save the updated dataset to a new CSV file
output_file_path = r'updated_combine_dataset X1X2X3.csv'
data.to_csv(output_file_path, index=False)

print(f"Updated dataset saved to {output_file_path}")