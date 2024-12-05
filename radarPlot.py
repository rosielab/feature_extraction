# Scaling values below for easier reading
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from sklearn.decomposition import PCA
import re
from sklearn.manifold import TSNE

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

# Replace 'saddness' affect
data[category_column] = data[category_column].replace('saddness', 'sadness')

# Group data by 'affect' and calculate the mean of the features
grouped_data = data.groupby(category_column)[features].mean().reset_index()

# Normalize feature values using StandardScaler
scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(grouped_data[features]),
    columns=features
)
normalized_data = pd.concat([grouped_data[category_column], normalized_features], axis=1)
print(normalized_data)

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

def create_radar_plot_emotion(data, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Number the axes
    axis_labels = list(range(1, len(categories) + 1))

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    unique_categories = data[category_column].unique()
    colors = plt.cm.tab10(range(len(unique_categories)))

    for i, row in data.iterrows():
        values = row.drop(category_column).values.flatten().tolist()
        values += values[:1]
        color = colors[list(unique_categories).index(row[category_column])]
        ax.fill(angles, values, alpha=0.25, color=color, label=row[category_column])
        ax.plot(angles, values, linewidth=2, color=color)

    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(-2.2, 2) # Adjust based on the data
    plt.title(title, size=16, y=1.1)
    
    # legend for emotions
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) 
    
    # legend for features
    legend_elements = [f"{i}: {category}" for i, category in enumerate(categories, 1)]
    legend_text = "\n".join(legend_elements)
    plt.gcf().text(0.8, 0.2, legend_text, fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.savefig(os.path.join(results_folder, filename),bbox_inches='tight')
    plt.close()

# Define different groupings for radar plots, can add more groups
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
    group_data = normalized_data[normalized_data[category_column].isin(emotions)]
    create_radar_plot_emotion(
        group_data, features, f"Radar Plot: {group_name}", f"radar_plot_{group_name.lower().replace(' ', '_')}.png"
    )

## Radar Plot for Phone Position Groups
'''
next to the body 1-7/19
1-2 m away 8-11/19
other side of the room 12-15/19
outside of the room 16-19/19
'''

# Function to extract the numeric value from the phone_position description
def extract_position_index(description):
    match = re.search(r'(\d+)/19', description)
    if match:
        return int(match.group(1))
    return None

# Apply preprocessing to extract numeric position index
data['phone_position_index'] = data['phone_position'].apply(extract_position_index)

# Classify phone positions into the defined groups
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
distance_group_data = data.groupby('distance_group')[features].mean().reset_index()
print (distance_group_data)
# Normalize the data for radar plotting
normalized_distance_features = pd.DataFrame(
    scaler.fit_transform(distance_group_data[features]),
    columns=features
)
normalized_distance_data = pd.concat([distance_group_data['distance_group'], normalized_distance_features], axis=1)
print(normalized_distance_data)

def create_radar_plot_distance(data, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] 

    # Number the axes
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
    ax.set_ylim(-2, 1.7) # Adjust based on the data
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Distance Groups")

    # legend for features
    legend_elements = [f"{i}: {category}" for i, category in enumerate(categories, 1)]
    legend_text = "\n".join(legend_elements)
    plt.gcf().text(0.8, 0.2, legend_text, fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.savefig(os.path.join(results_folder, filename), bbox_inches='tight')
    plt.close()

create_radar_plot_distance(
    normalized_distance_data,
    features,
    "Radar Plot of Features by Phone Position Groups",
    "radar_plot_distance_groups.png"
)

# Function to create radar plot for a single emotion
def create_radar_distance_per_emotion(data, emotion, categories, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Number the axes
    axis_labels = list(range(1, len(categories) + 1))

    # Sorting the groups to control legend order
    sorted_data = data.sort_values(by='distance_group', key=lambda x: x.map({
        'Next to Body': 1,
        '1-2 m Away': 2,
        'Other Side of Room': 3,
        'Outside of Room': 4
    }))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(range(len(data)))

    for i, row in sorted_data.iterrows():
        values = row.drop('distance_group').values.flatten().tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[i], label=row['distance_group'])
        ax.plot(angles, values, linewidth=2, color=colors[i])

    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(-2, 2)  # Adjust based on the data
    plt.title(f"{title} - {emotion}", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Distance Groups")
    
    # legend for features
    legend_elements = [f"{i}: {category}" for i, category in enumerate(categories, 1)]
    legend_text = "\n".join(legend_elements)
    plt.gcf().text(0.8, 0.2, legend_text, fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.savefig(os.path.join(results_folder, filename), bbox_inches='tight')
    plt.close()

unique_emotions = data[category_column].unique()

for emotion in unique_emotions:
    # Filter data for the specific emotion
    emotion_data = data[data[category_column] == emotion]
    
    # Group by distance_group and calculate mean feature values
    emotion_distance_group_data = emotion_data.groupby('distance_group')[features].mean().reset_index()

    # Normalize the data for radar plotting
    normalized_emotion_features = pd.DataFrame(
        scaler.fit_transform(emotion_distance_group_data[features]),
        columns=features
    )
    normalized_emotion_data = pd.concat([emotion_distance_group_data['distance_group'], normalized_emotion_features], axis=1)

    # Create radar plot for the emotion
    create_radar_distance_per_emotion(
        normalized_emotion_data,
        emotion,
        features,
        "Radar Plot of Features by Phone Position",
        f"radar_plot_distance_{emotion.lower().replace(' ', '_')}.png"
    )


# Define a function for scatter plot
def scatter_plot_gemaps(data, features, title, filename):
    # Perform PCA for dimensionality reduction (to 2D for scatter plot)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[features])
    data['PC1'] = pca_result[:, 0]
    data['PC2'] = pca_result[:, 1]

    # Facet the scatter plot by 'distance_group'
    g = sns.FacetGrid(data, col="distance_group", hue=category_column, palette="tab10", height=4, col_wrap=2)
    g.map(sns.scatterplot, "PC1", "PC2", alpha=0.7)

    # Add legend
    g.add_legend()

    # Save the faceted plots
    plt.savefig(os.path.join(results_folder, 'facet_scatter_by_distance_group.png'), bbox_inches='tight')
    plt.close()
    
    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue=category_column,  # Color by emotion
        style='distance_group',  # Shape by distance
        data=data,
        palette='tab10',
        alpha=0.7,
        s=100  # Marker size
    )

    # Add title and labels
    plt.title(title, size=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title="Emotion & Distance", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), bbox_inches='tight')
    plt.close()

# Call the function
scatter_plot_gemaps(
    data, 
    features, 
    "Scatter Plot of GeMAPS Features (Colored by Emotion, Shaped by Distance)", 
    "scatter_plot_gemaps_emotion_distance.png"
)

def create_radar_emotions_per_distance(data, categories, distance_group, title, filename):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle

    # Number the axes
    axis_labels = list(range(1, len(categories) + 1))

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    unique_categories = data[category_column].unique()
    colors = plt.cm.tab10(range(len(unique_categories)))

    for i, row in data.iterrows():
        values = row.drop(category_column).values.flatten().tolist()
        values += values[:1]
        color = colors[list(unique_categories).index(row[category_column])]
        ax.fill(angles, values, alpha=0.25, color=color, label=row[category_column])
        ax.plot(angles, values, linewidth=2, color=color)

    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(-2.2, 2)  # Adjust based on the data
    plt.title(f"{title} - {distance_group}", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Emotions")

    # legend for features
    legend_elements = [f"{i}: {category}" for i, category in enumerate(categories, 1)]
    legend_text = "\n".join(legend_elements)
    plt.gcf().text(0.8, 0.2, legend_text, fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.savefig(os.path.join(results_folder, filename), bbox_inches='tight')
    plt.close()

# Loop through each distance group and create a radar plot for each
distance_groups = data['distance_group'].unique()

for group in distance_groups:
    # Filter data for the specific distance group
    group_data = data[data['distance_group'] == group]

    # High Arousal Emotions
    high_arousal_data = group_data[group_data[category_column].isin(groupings['High Arousal'])]
    if not high_arousal_data.empty:
        # Group by emotion and calculate mean feature values
        high_arousal_grouped = high_arousal_data.groupby(category_column)[features].mean().reset_index()
        
        # Normalize the data for radar plotting
        normalized_high_arousal_features = pd.DataFrame(
            scaler.fit_transform(high_arousal_grouped[features]),
            columns=features
        )
        normalized_high_arousal_data = pd.concat(
            [high_arousal_grouped[category_column], normalized_high_arousal_features], axis=1
        )

        # Create radar plot for high arousal
        create_radar_emotions_per_distance(
            normalized_high_arousal_data,
            features,
            group,
            "Radar Plot of Features by High Arousal Emotions",
            f"radar_plot_high_arousal_{group.lower().replace(' ', '_')}.png"
        )

    # Low Arousal Emotions
    low_arousal_data = group_data[group_data[category_column].isin(groupings['Low Arousal'])]
    if not low_arousal_data.empty:
        # Group by emotion and calculate mean feature values
        low_arousal_grouped = low_arousal_data.groupby(category_column)[features].mean().reset_index()
        
        # Normalize the data for radar plotting
        normalized_low_arousal_features = pd.DataFrame(
            scaler.fit_transform(low_arousal_grouped[features]),
            columns=features
        )
        normalized_low_arousal_data = pd.concat(
            [low_arousal_grouped[category_column], normalized_low_arousal_features], axis=1
        )

        # Create radar plot for low arousal
        create_radar_emotions_per_distance(
            normalized_low_arousal_data,
            features,
            group,
            "Radar Plot of Features by Low Arousal Emotions",
            f"radar_plot_low_arousal_{group.lower().replace(' ', '_')}.png"
        )