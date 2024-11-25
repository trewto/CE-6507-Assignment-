#

import numpy as np
import seaborn as sns
import pandas as pd

# Import CSV file
csv_file_path = r'D:\MSc\T_1\CE 6507  Traffic Engineering (Mozzem Sir)\Assignment\Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe
print(data.head())


# Filter the dataset for the specified conditions
filtered_data = data[(data['Lane_ID'] == 2) & (data['v_Class'] == 2) & (data['Location'] == "us-101") & (data['Following'] != 0)]


# Remove duplicate rows
filtered_data_no_duplicates = filtered_data.drop_duplicates()

# Select the top 500 rows from the deduplicated data
filtered_top_500 = filtered_data_no_duplicates.head(1000)

# Save the filtered dataset to a new CSV file
output_file_path = r'dataset.csv'
filtered_top_500.to_csv(output_file_path, index=False)

print(f"Filtered dataset (top  rows) without duplicates saved to {output_file_path}")