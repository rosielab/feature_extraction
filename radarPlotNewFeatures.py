import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import re
from math import pi
from sklearn.manifold import TSNE

# Define file paths
DATA_CSV_FILE = './updated_toolbox.csv'  # Replace with your actual path
RESULTS_FOLDER = './resultsNewFeats'  # Replace with your actual path

# Load data
data = pd.read_csv(DATA_CSV_FILE)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Features and category column
FEATURES = [
    'mean pitch', 'pitch range', 'speech rate', 'energy', 'shimmer', 
    'jitter', 'spectral slope', 'mean intensity', 'max intensity'
]
CATEGORY_COLUMN = 'emotion'

# Replace 'saddness' with 'sadness' in the category column
data[CATEGORY_COLUMN] = data[CATEGORY_COLUMN].replace('saddness', 'sadness')

# Group data by emotion and calculate mean of features
grouped_data = data.groupby(CATEGORY_COLUMN)[FEATURES].mean().reset_index()

# Normalize the features using StandardScaler
scaler = StandardScaler()
normalized_features = pd.DataFrame(scaler.fit_transform(grouped_data[FEATURES]), columns=FEATURES)
normalized_data = pd.concat([grouped_data[CATEGORY_COLUMN], normalized_features], axis=1)
print(normalized_data)

# Function to plot correlation heatmap
def plot_correlation_heatmap(data, features, filename):
    corr_matrix = data[features].corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Features")
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Plot correlation heatmap
plot_correlation_heatmap(data, FEATURES, 'correlation_heatmap.png')

# Function to plot mean features across affect categories
def plot_mean_features(normalized_data, category_column, filename):
    mean_data = normalized_data.groupby(category_column).mean()
    plt.figure(figsize=(16, 10))
    mean_data.T.plot(marker='o', figsize=(16, 6))
    plt.title("Line Plot of Feature Means by Affect Category")
    plt.ylabel("Normalized Feature Value")
    plt.legend(title="Affect Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Plot line plot of mean features
plot_mean_features(normalized_data, CATEGORY_COLUMN, 'line_plot_features.png')

# Function to plot heatmap of feature averages by affect category
def plot_average_feature_heatmap(normalized_data, category_column, filename):
    average_data = normalized_data.groupby(category_column).mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(average_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Heatmap of Average Feature Values by Affect Category")
    plt.xlabel("Features")
    plt.ylabel("Affect Categories")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Plot heatmap of feature averages
plot_average_feature_heatmap(normalized_data, CATEGORY_COLUMN, 'heatmap_feature_averages.png')

# Function to create radar plot for emotion data
def create_radar_plot_emotion(data, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Number the axes
    axis_labels = list(range(1, len(categories) + 1))

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    unique_categories = data[CATEGORY_COLUMN].unique()
    colors = plt.cm.tab10(range(len(unique_categories)))

    for i, row in data.iterrows():
        values = row.drop(CATEGORY_COLUMN).values.flatten().tolist()
        values += values[:1]
        color = colors[list(unique_categories).index(row[CATEGORY_COLUMN])]
        ax.fill(angles, values, alpha=0.25, color=color, label=row[CATEGORY_COLUMN])
        ax.plot(angles, values, linewidth=2, color=color)

    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(-2.2, 2)  # Adjust based on the data
    plt.title(title, size=16, y=1.1)
    
    # legend for emotions
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) 
    
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Define emotion groupings
groupings = {
    "Positive Emotions": ["joy", "surprise"],
    "Negative Emotions": ["anger", "fear", "disgust", "sadness"],
    "High Arousal": ["anger", "surprise", "fear", "joy"],
    "Low Arousal": ["neutral", "sadness", "disgust"],
    "Neutral Emotion": ["neutral"],
    "Mixed Group 1": ["joy", "fear", "neutral"],
    "Mixed Group 2": ["anger", "surprise", "disgust"]
}

# Generate radar plots for each grouping
for group_name, emotions in groupings.items():
    group_data = normalized_data[normalized_data[CATEGORY_COLUMN].isin(emotions)]
    create_radar_plot_emotion(group_data, FEATURES, f"Radar Plot: {group_name}", f"radar_plot_{group_name.lower().replace(' ', '_')}.png")

# Function to extract numeric value from phone position description
def extract_position_index(description):
    if isinstance(description, str):  # Ensure the description is a string
        match = re.search(r'(\d+)/19', description)
        if match:
            return int(match.group(1))
    return None

# Apply preprocessing to extract numeric position index
data['phone_position_index'] = data['phone_position'].apply(extract_position_index)

# Classify phone positions into defined groups
def classify_position(position):
    if position in range(1, 8):
        return "Next to Body"
    elif position in range(8, 12):
        return "1-2 m Away"
    elif position in range(12, 16):
        return "Other Side of Room"
    elif position in range(16, 20):
        return "Outside of Room"
    return "Unknown"

data['distance_group'] = data['phone_position_index'].apply(classify_position)

# Group by the new distance_group column and calculate mean feature values
distance_group_data = data.groupby('distance_group')[FEATURES].mean().reset_index()

# Normalize the data for radar plotting
normalized_distance_features = pd.DataFrame(scaler.fit_transform(distance_group_data[FEATURES]), columns=FEATURES)
normalized_distance_data = pd.concat([distance_group_data['distance_group'], normalized_distance_features], axis=1)

# Function to create radar plot for distance groups
def create_radar_plot_distance(data, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle

    axis_labels = list(range(1, len(categories) + 1))

    # Sorting the groups to control legend order
    sorted_data = data.sort_values(by='distance_group', key=lambda x: x.map({
        'Next to Body': 1,
        '1-2 m Away': 2,
        'Other Side of Room': 3,
        'Outside of Room': 4
    }))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(range(len(sorted_data)))

    for i, row in sorted_data.iterrows():
        values = row.drop('distance_group').values.flatten().tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[i], label=row['distance_group'])
        ax.plot(angles, values, linewidth=2, color=colors[i])

    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(-2, 1.7)  # Adjust based on the data
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Distance Groups")

    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Create radar plot for distance groups
create_radar_plot_distance(normalized_distance_data, FEATURES, "Radar Plot of Features by Phone Position Groups", "radar_plot_distance_groups.png")

# Scatter plot for PCA
def scatter_plot_pca(data, features, title, filename):
    # Perform PCA for dimensionality reduction (to 2D for scatter plot)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[features])
    data['PC1'] = pca_result[:, 0]
    data['PC2'] = pca_result[:, 1]

    # Create a scatter plot of PCA results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue=data[CATEGORY_COLUMN], palette='viridis', data=data)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title="Emotion Categories", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# Scatter plot using PCA
scatter_plot_pca(normalized_data, FEATURES, "PCA Scatter Plot of Features", "pca_scatter_plot.png")

# Function for t-SNE visualization
def tsne_plot(data, features, title, filename):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(data[features])
    data['t-SNE1'] = tsne_result[:, 0]
    data['t-SNE2'] = tsne_result[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue=data[CATEGORY_COLUMN], palette='viridis', data=data)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename), bbox_inches='tight')
    plt.close()

# t-SNE plot
#tsne_plot(normalized_data, FEATURES, "t-SNE Visualization of Features", "tsne_plot.png")
