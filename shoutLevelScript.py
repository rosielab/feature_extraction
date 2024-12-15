import pandas as pd

# Load the CSV files
csv1 = pd.read_csv('./tidied_dataShoutLevels.csv')
csv2 = pd.read_csv('./toolbox_features.csv')

# Clean and standardize the 'shout_level' column
csv1['shout_level'] = csv1['shout_level'].str.strip().str.lower()

# Map shout levels to numeric values
shout_mapping = {"shout": "1", "no-shout": "0", "n/a": "1",} # n/a is shout because it's likely i mistakenly put it as it's under the shout
csv1['shout'] = csv1['shout_level'].map(shout_mapping)

# Check for null values in the mapping
if csv1['shout'].isnull().any():
    print("Unmapped shout_level values:", csv1.loc[csv1['shout'].isnull(), 'shout_level'].unique())

# Clean the 'file_location' column in both CSVs for matching
csv1['file_location'] = csv1['file_location'].str.strip()
csv2['file_location'] = csv2['file_location'].str.strip()

# Merge CSVs on 'file_location'
csv2 = pd.merge(csv2, csv1[['file_location', 'shout', 'affect', 'phone_position']], on='file_location', how='left')

# Save the updated CSV2
csv2.to_csv('updated_csv2.csv', index=False)

# Check for missing matches
missing_matches = csv2[csv2['shout'].isnull()]
if not missing_matches.empty:
    print("Unmatched file_location values in CSV2:", missing_matches['file_location'].unique())