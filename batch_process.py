import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from parsel_process import analyse_pitch, analyse_formants, analyse_mfcc, get_energy, analyse_intensity, get_max_intensity, analyse_zero_crossing, pauses, spectral_slope, analyse_harmonics, mean_spectral_rolloff, get_number_words, get_number_sylls, cleaning, get_envelope, analyse_jitter, analyse_shimmer

# Set up paths
CSV_FILE = './tidied_data.csv'
RESULTS_PATH = './results/'

# Function to extract features using parsel_process.py functions
def extract_features(filepath):
    features = {}
    features['Pitch'] = analyse_pitch(filepath)
    features['Formants_f1'] = analyse_formants(1, filepath)
    features['Formants_f2'] = analyse_formants(2, filepath)
    features['MFCC_mean'] = analyse_mfcc(filepath, RESULTS_PATH)
    features['Energy'] = get_energy(filepath)
    features['Intensity'] = analyse_intensity(filepath)
    features['Max_Intensity'] = get_max_intensity(filepath)
    features['Zero_Crossings'] = analyse_zero_crossing(filepath)
    features['Pauses'] = pauses(filepath)
    features['Spectral_Slope'] = spectral_slope(filepath)
    features['Harmonics_to_Noise'] = analyse_harmonics(filepath)
    features['Mean_Spectral_Rolloff'] = mean_spectral_rolloff(filepath)
    features['Number_of_Words'] = get_number_words(filepath)
    features['Number_of_Syllables'] = get_number_sylls(filepath)
    #features['Cleaning'] = cleaning(filepath)
    features['Envelope'] = get_envelope(filepath)
    features['Jitter'] = analyse_jitter(filepath)
    features['Shimmer'] = analyse_shimmer(filepath)
    return features

# Load the CSV file
data_df = pd.read_csv(CSV_FILE)

# Initialize lists to store features and labels
feature_vectors = []
emotion_labels = []

# Iterate through rows in the CSV to access .wav files and extract features
for index, row in data_df.iterrows():
    file_location = os.path.join("C:\\Users\\12364\\Desktop\\feature_extraction\\", row['file_location'])
    #print(file_location)
    affect = row['affect']
    if file_location.endswith(".wav") and os.path.exists(file_location):
        features = extract_features(file_location)
        feature_vectors.append(list(features.values()))
        emotion_labels.append(affect)

print("Feature Vectors:", feature_vectors)

# Convert lists to numpy arrays for clustering
X = np.array(feature_vectors)
y = np.array(emotion_labels)

# Perform clustering (example using KMeans)
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(X)

# Create a folder to store clustering results if it doesn't exist
cluster_results_path = os.path.join(RESULTS_PATH, 'clusters/')
os.makedirs(cluster_results_path, exist_ok=True)

# Save cluster results to a CSV file
results_df = pd.DataFrame({'Filename': data_df['file_location'], 'Affect': emotion_labels, 'Cluster': clusters})
results_df.to_csv(os.path.join(cluster_results_path, 'cluster_results.csv'), index=False)

# Plotting features with color coding based on 'affect'
plt.figure(figsize=(10, 6))
unique_affects = list(set(y))
colors = plt.cm.get_cmap('viridis', len(unique_affects))

# Example plotting based on MFCC mean and Energy, color-coded by 'Affect'
for affect in unique_affects:
    idx = np.where(y == affect)
    plt.scatter(X[idx, 3], X[idx, 4], c=[colors(unique_affects.index(affect))], label=affect)

plt.xlabel('MFCC Mean')
plt.ylabel('Energy')
plt.title('Clustering and Affect-Based Color Coding')
plt.colorbar(ticks=range(len(unique_affects)), label='Affect')
plt.legend()
plt.savefig(os.path.join(RESULTS_PATH, 'visualizations/affect_mfcc_energy_cluster.png'))
plt.show()
