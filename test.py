# import os
# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from subprocess import call

# # Configuration
# input_folder = 'path/to/your/audio/files'
# output_folder = 'path/to/output/features'
# csv_file = 'path/to/labels.csv'
# n_clusters = 3  # Number of clusters for KMeans

# # Load CSV file
# df = pd.read_csv(csv_file)

# # Function to extract features using Voice Toolbox
# def extract_features(audio_path, output_path):
#     call([
#         'python3', 'parsel_process.py', '16000', audio_path, output_path,
#         '--formants', '--ZCR', '--harmonics', '--rate_of_speech',
#         '--loudness', '--pitch_features', '--spectral_features', '--energy'
#     ])

# # Extract features for each audio file
# for index, row in df.iterrows():
#     audio_path = os.path.join(input_folder, row['filename'])
#     output_path = os.path.join(output_folder, f"{index}.csv")
#     extract_features(audio_path, output_path)

# # Combine all extracted features into a single DataFrame
# feature_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.csv')]
# features = [pd.read_csv(f) for f in feature_files]
# features_df = pd.concat(features, ignore_index=True)

# # Merge features with emotion labels
# features_df['emotion'] = df['emotion']

# # Perform clustering on features
# X = features_df.drop(columns=['emotion'])
# kmeans = KMeans(n_clusters=n_clusters)
# features_df['cluster'] = kmeans.fit_predict(X)

# # Save clustering results
# features_df.to_csv('clustering_results.csv', index=False)

# # Plot features
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=features_df, x='feature1', y='feature2', hue='cluster', style='emotion', palette='viridis')
# plt.title('Feature Clustering')
# plt.savefig('feature_clustering.png')
# plt.show()

# # Plot individual feature distributions
# for column in X.columns:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data=features_df, x=column, hue='emotion', kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.savefig(f'distribution_{column}.png')
#     plt.show()

# # Optional: Listen to and label subgroups
# # (Implement a mechanism to play audio files and update labels as needed)

# # Plot audio features (e.g., using radar plots)
# # Example: Radar plot for mean features of each cluster
# import matplotlib.pyplot as plt
# import numpy as np

# # Radar plot function
# def radar_plot(data, categories, group_names):
#     N = len(categories)
#     angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
#     data = np.concatenate((data, data[:, [0]]), axis=1)
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     for i, group in enumerate(group_names):
#         ax.fill(angles, data[i], alpha=0.25, label=group)
#         ax.plot(angles, data[i], linewidth=2, linestyle='solid')

#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories)
#     plt.legend(loc='upper right')
#     plt.title('Mean Features of Clusters')
#     plt.show()

# # Prepare data for radar plot
# grouped = features_df.groupby('cluster').mean().drop(columns=['cluster', 'emotion'])
# categories = list(grouped.columns)
# data = grouped.to_numpy()

# # Plot radar chart
# radar_plot(data, categories, [f'Cluster {i}' for i in range(n_clusters)])
###################################################################################################################
# import os
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from parsel_process import analyse_pitch, analyse_formants, analyse_mfcc, get_energy, analyse_intensity, get_max_intensity, analyse_zero_crossing, pauses, spectral_slope, analyse_harmonics, mean_spectral_rolloff, get_number_words, get_number_sylls, cleaning, get_envelope, analyse_jitter, analyse_shimmer

# # Set up paths
# DATA_PATH = './data/audio_files/'  # Adjust this path according to your actual folder structure
# LABELS_FILE = './data/labels.csv'  # Adjust this path according to your actual folder structure
# RESULTS_PATH = './results/'

# # Function to extract features using parsel_process.py functions
# def extract_features(filepath):
#     features = {}
#     features['Pitch'] = analyse_pitch(filepath)
#     features['Formants_f1'] = analyse_formants(1, filepath)
#     features['Formants_f2'] = analyse_formants(2, filepath)
#     features['MFCC_mean'] = analyse_mfcc(filepath, RESULTS_PATH)
#     features['Energy'] = get_energy(filepath)
#     features['Intensity'] = analyse_intensity(filepath)
#     features['Max_Intensity'] = get_max_intensity(filepath)
#     features['Zero_Crossings'] = analyse_zero_crossing(filepath)
#     features['Pauses'] = pauses(filepath)
#     features['Spectral_Slope'] = spectral_slope(filepath)
#     features['Harmonics_to_Noise'] = analyse_harmonics(filepath)
#     features['Mean_Spectral_Rolloff'] = mean_spectral_rolloff(filepath)
#     features['Number_of_Words'] = get_number_words(filepath)
#     features['Number_of_Syllables'] = get_number_sylls(filepath)
#     features['Cleaning'] = cleaning(filepath)
#     features['Envelope'] = get_envelope(filepath)
#     features['Jitter'] = analyse_jitter(filepath)
#     features['Shimmer'] = analyse_shimmer(filepath)
#     return features

# # Load labels CSV
# labels_df = pd.read_csv(LABELS_FILE)

# # Initialize lists to store features and labels
# feature_vectors = []
# emotion_labels = []

# # Iterate through audio files and extract features
# for filename in os.listdir(DATA_PATH):
#     if filename.endswith(".wav"):
#         filepath = os.path.join(DATA_PATH, filename)
#         emotion_label = labels_df.loc[labels_df['Filename'] == filename]['Emotion'].values[0]
#         features = extract_features(filepath)
#         feature_vectors.append(list(features.values()))
#         emotion_labels.append(emotion_label)

# # Convert lists to numpy arrays for clustering
# X = np.array(feature_vectors)
# y = np.array(emotion_labels)

# # Perform clustering (example using KMeans)
# kmeans = KMeans(n_clusters=4, random_state=0)
# clusters = kmeans.fit_predict(X)

# # Create a folder to store clustering results if it doesn't exist
# cluster_results_path = os.path.join(RESULTS_PATH, 'clusters/')
# os.makedirs(cluster_results_path, exist_ok=True)

# # Save cluster results to a CSV file
# results_df = pd.DataFrame({'Filename': os.listdir(DATA_PATH), 'Emotion': emotion_labels, 'Cluster': clusters})
# results_df.to_csv(os.path.join(cluster_results_path, 'cluster_results.csv'), index=False)

# # Example of plotting a feature (MFCC mean) for illustration
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 3], X[:, 4], c=clusters, cmap='viridis')
# plt.xlabel('MFCC Mean')
# plt.ylabel('Energy')
# plt.title('Clustering based on MFCC Mean and Energy')
# plt.colorbar(label='Cluster')
# plt.savefig(os.path.join(RESULTS_PATH, 'visualizations/mfcc_energy_cluster.png'))
# plt.show()



# shapes for obstruction vs non obstruction
# colors for the distance
# colors for emotions as well or diferent radar plots