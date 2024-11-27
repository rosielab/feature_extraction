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

file = 'C:\\Users\\12364\\Desktop\\feature_extraction\\shout-data\\shout-data\\shout_data_00a0fb59-49f0-40e1-8481-6a334c65b126\\final\\chunk11.wav'
signal, sampling_rate = audiofile.read(
    file,
    always_2d=True  # Keep this to ensure the output is always 2D
)

smile = opensmile.Smile(
    # feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_set=opensmile.FeatureSet.eGeMAPSv01a,
    # feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
) 
smile.feature_names
['Loudness_sma3',
 'alphaRatio_sma3',
 'hammarbergIndex_sma3',
 'slope0-500_sma3',
 'slope500-1500_sma3',
 'spectralFlux_sma3',
 'mfcc1_sma3',
 'mfcc2_sma3',
 'mfcc3_sma3',
 'mfcc4_sma3',
 'F0semitoneFrom27.5Hz_sma3nz',
 'jitterLocal_sma3nz',
 'shimmerLocaldB_sma3nz',
 'HNRdBACF_sma3nz',
 'logRelF0-H1-H2_sma3nz',
 'logRelF0-H1-A3_sma3nz',
 'F1frequency_sma3nz',
 'F1bandwidth_sma3nz',
 'F1amplitudeLogRelF0_sma3nz',
 'F2frequency_sma3nz',
 'F2bandwidth_sma3nz',
 'F2amplitudeLogRelF0_sma3nz',
 'F3frequency_sma3nz',
 'F3bandwidth_sma3nz',
 'F3amplitudeLogRelF0_sma3nz']

print("processing file now using GeMAPS\n")
y = smile.process_signal(
    signal,
    sampling_rate
)
print("done processing, will print now\n")


y.to_csv('C:\\Users\\12364\Desktop\\feature_extraction\\processed_features.csv', index=False)

# printing the columns in the GeMAPS processing
for col in y.columns:
    print(col)

print("\nCalculating all the ranges\n")

range_values = y.max(axis=0) - y.min(axis=0)
print(range_values)

############################################################
print("\nCalculating all the means\n")

# Compute range for all columns
means = y.mean(axis=0)
print(means)

# print(smile.feature_set)
