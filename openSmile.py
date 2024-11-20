# Feature Extraction using OpenSmile
# Docs: https://audeering.github.io/opensmile-python/ 
# Feature Set Documentation: https://audeering.github.io/opensmile-python/api/opensmile.FeatureSet.html
# GeMAPS Feature Set Documentation: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf
# OpenSmile Python Package Github: https://github.com/audeering/opensmile-python 
# OpenSmile citation: Florian Eyben, Martin Wöllmer, Björn Schuller: “openSMILE - The Munich Versatile and Fast Open-Source Audio Feature Extractor”, Proc. ACM Multimedia (MM), ACM, Florence, Italy, ISBN 978-1-60558-933-6, pp. 1459-1462, 25.-29.10.2010.
# Step 1: pip install opensmile --user
# Step 2: python openSmile.py

import opensmile


smile = opensmile.Smile(
    # feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print("processing file now using GeMAPS")
y = smile.process_file('C:\\Users\\12364\Desktop\\feature_extraction\\shout-data\\shout-data\\shout_data_00a0fb59-49f0-40e1-8481-6a334c65b126\\final\\chunk11.wav')
print("done processing, will print now")
# print(y)

# printing the columns in the GeMAPS processing
for col in y.columns:
    print(col)

print("calculating all the means")

means = y.mean(axis=0)  # Compute mean for all columns
print(means)

# print(smile.feature_set)