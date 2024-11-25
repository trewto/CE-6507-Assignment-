import pandas as pd

# Import the dataset
csv_file_path = r'dataset.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe
print("Original Dataset:")
print(data.head())

# Display all column names
print("Column names in the dataset:")
print(data.columns.tolist())

# Initialize a list to store the combined rows
combined_rows = []

# Iterate through each row in the dataset
for _, row in data.iterrows():
    following_vehicle_id = row['Following']
    
    # Check if the Following ID exists in the Vehicle_ID column
    leader_row = data[data['Vehicle_ID'] == following_vehicle_id]
    
    if not leader_row.empty:
        # Merge the follower and leader row, adding leader data to the follower row
        leader_info = leader_row.iloc[0]
        
        # Prefix the leader columns with 'leader_'
        leader_info_prefixed = leader_info.add_prefix('leader_')
        
        # Combine the original row with the leader data
        combined_row = pd.concat([row, leader_info_prefixed])
        
        # Add the combined row to the list
        combined_rows.append(combined_row)

# Convert the combined rows list into a DataFrame
combined_data = pd.DataFrame(combined_rows)

# Ensure that the column names are preserved and the "leader_" columns are correctly prefixed
# Preserve original column names and add "leader_" for leader data
original_columns = data.columns.tolist()  # Get original column names
leader_columns = [f"leader_{col}" for col in original_columns]  # Add "leader_" to original column names for leader data

# Set the correct column names for the combined dataset
combined_data.columns = original_columns + leader_columns

# Display the combined dataset
print("Combined Dataset (Follower and Leader):")
print(combined_data.head())


print("Column names in the dataset:")

print(combined_data.columns.tolist())
## Save the combined dataset to a new CSV file
#output_file_path = r'combined_dataset.csv'
#combined_data.to_csv(output_file_path, index=False)
#print(f"Combined dataset saved to {output_file_path}")
