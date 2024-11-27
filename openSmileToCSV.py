# Feature Extraction using OpenSmile
# Docs: https://audeering.github.io/opensmile-python/ 
# Feature Set Documentation: https://audeering.github.io/opensmile-python/api/opensmile.FeatureSet.html
# GeMAPS Feature Set Documentation: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf
# OpenSmile Python Package Github: https://github.com/audeering/opensmile-python 
# OpenSmile citation: Florian Eyben, Martin Wöllmer, Björn Schuller: “openSMILE - The Munich Versatile and Fast Open-Source Audio Feature Extractor”, Proc. ACM Multimedia (MM), ACM, Florence, Italy, ISBN 978-1-60558-933-6, pp. 1459-1462, 25.-29.10.2010.
# Step 1: pip install opensmile --user
# Step 2: python openSmile.py

import os
import time

import numpy as np
import pandas as pd

import audiofile
import opensmile

import os
import pandas as pd
import audiofile
import opensmile

# Input and output CSV paths
INPUT_CSV_FILE = './tidied_data.csv'
OUTPUT_CSV_FILE = './processed_data_with_features.csv'

# Load the input CSV
data_df = pd.read_csv(INPUT_CSV_FILE)

# Slice the first 10 rows for testing
data_df = data_df.head(10)  # This selects the first 10 rows

# Create an empty list to hold new rows with additional features
new_rows = []

# Iterate through each row in the input CSV
for index, row in data_df.iterrows():
    file = os.path.join("C:\\Users\\12364\\Desktop\\feature_extraction\\", row['file_location'])
    
    if file.endswith(".wav") and os.path.exists(file):
        # Read the audio file
        signal, sampling_rate = audiofile.read(
            file,
            always_2d=True  # Ensure the output is always 2D
        )

        # Initialize OpenSMILE
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        
        # Process the signal to extract features
        y = smile.process_signal(signal, sampling_rate)

        # Calculate required statistics
        try:
            f1_range = y['F1frequency_sma3nz'].max() - y['F1frequency_sma3nz'].min()
            slope0_mean = y['slope0-500_sma3'].mean()
            slope500_mean = y['slope500-1500_sma3'].mean()
            f1_mean = y['F1frequency_sma3nz'].mean()
        except KeyError:
            # If columns are missing, handle gracefully
            f1_range = slope0_mean = slope500_mean = f1_mean = None

        # Create a new row by appending the additional features
        new_row = row.to_dict()
        new_row['F1frequency_range'] = f1_range
        new_row['slope0-500_mean'] = slope0_mean
        new_row['slope500-1500_mean'] = slope500_mean
        new_row['F1frequency_mean'] = f1_mean
        new_rows.append(new_row)

# Convert the new rows into a DataFrame
processed_df = pd.DataFrame(new_rows)

# Save the updated DataFrame to a new CSV
processed_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"Processed data saved to {OUTPUT_CSV_FILE}")
