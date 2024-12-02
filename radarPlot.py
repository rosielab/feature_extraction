# Scaling values below for easier reading
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from sklearn.decomposition import PCA

# Load the CSV file
DATA_CSV_FILE = './processed_data_with_features_2.csv'  # Replace with your actual path
results_folder = './results'  # Replace with your actual path
data = pd.read_csv(DATA_CSV_FILE)
os.makedirs(results_folder, exist_ok=True)

# Features and category column
features = [
    'F1frequency_range', 
    'slope0-500_mean', 
    'slope500-1500_mean', 
    'F1frequency_mean', 
    'shimmer_mean', 
    'jitter_mean', 
    'loudness_mean'
]
category_column = 'affect'

# Filter out 'sadness' affect
data[category_column] = data[category_column].replace('saddness', 'sadness')

# Split data into two groups for radar plots
group_1_acts = ['fear', 'disgust', 'anger', 'sadness']  # Group 1 (first radar plot)
group_2_acts = ['neutral', 'joy', 'surprise']  # Group 2 (second radar plot)

# Group data by 'affect' and calculate the mean of the features
grouped_data = data.groupby(category_column)[features].mean().reset_index()

# Normalize feature values using MinMaxScaler
scaler = MinMaxScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(grouped_data[features]),
    columns=features
)
normalized_data = pd.concat([grouped_data[category_column], normalized_features], axis=1)

# Correlation heatmap
corr_matrix = data[features].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.savefig(os.path.join(results_folder, 'correlation_heatmap.png'),bbox_inches='tight')
plt.close()

# Line plot for means of features across affect categories
mean_data = normalized_data.groupby(category_column).mean()
plt.figure(figsize=(16, 10))
mean_data.T.plot(marker='o', figsize=(16, 6))
plt.title("Line Plot of Feature Means by Affect Category")
plt.ylabel("Normalized Feature Value")
plt.legend(title="Affect Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.savefig(os.path.join(results_folder, 'line_plot_features.png'),bbox_inches='tight')
plt.close()

# Heatmap of Feature Averages by Affect Category
average_data = normalized_data.groupby(category_column).mean()
plt.figure(figsize=(12, 8))
sns.heatmap(average_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of Average Feature Values by Affect Category")
plt.xlabel("Features")
plt.ylabel("Affect Categories")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'heatmap_feature_averages.png'), bbox_inches='tight')
plt.close()

def create_and_save_radar_plot(data, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    unique_categories = data[category_column].unique()
    colors = plt.cm.tab10(range(len(unique_categories)))

    for i, row in data.iterrows():
        values = row.drop(category_column).values.flatten().tolist()
        values += values[:1]
        color = colors[list(unique_categories).index(row[category_column])]
        ax.fill(angles, values, alpha=0.25, color=color, label=row[category_column])
        ax.plot(angles, values, linewidth=2, color=color)

    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(0, 1)
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(results_folder, filename),bbox_inches='tight')
    plt.close()

# Define different groupings for radar plots, can add more groups
groupings = {
    "Positive Emotions": ["joy", "surprise"],
    "Negative Emotions": ["anger", "fear", "disgust", "sadness"],
    "High Arousal": ["anger", "surprise", "fear"],
    "Low Arousal": ["neutral", "sadness", "disgust"],
    "Neutral Emotion": ["neutral"],
    "Mixed Group 1": ["joy", "fear", "neutral"],
    "Mixed Group 2": ["anger", "surprise", "disgust"]
}

# Generate radar plots for each grouping
for group_name, emotions in groupings.items():
    group_data = normalized_data[normalized_data[category_column].isin(emotions)]
    create_and_save_radar_plot(
        group_data, features, f"Radar Plot: {group_name}", f"radar_plot_{group_name.lower().replace(' ', '_')}.png"
    )


