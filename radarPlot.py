# Scaling values below for easier reading
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import re

# Load the CSV file
same_scale_results_folder = './results_same_scale'
diff_scale_results_folder = './results_diff_scale'
data_features = pd.read_csv('./toolbox_features.csv')
data_shout_level = pd.read_csv('./tidied_data_shoutLevels.csv')
data = pd.merge(data_features, data_shout_level, on='file_location')
os.makedirs(same_scale_results_folder, exist_ok=True)
os.makedirs(diff_scale_results_folder, exist_ok=True)

features = [
    'mean pitch', 
    'pitch range', 
    'speech rate', 
    'energy', 
    'shimmer', 
    'jitter', 
    'spectral slope',
    'mean intensity',
    'max intensity'
]
category_column = 'affect'

data[category_column] = data[category_column].replace('saddness', 'sadness')

def extract_position_index(description):
    match = re.search(r'(\d+)/19', description)
    if match:
        return int(match.group(1))
    return None

data['phone_position_index'] = data['phone_position'].apply(extract_position_index)

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

# Normalize the entire dataset
scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(data[features]),
    columns=features
)

# Combine normalized features with non-feature columns
normalized_data = pd.concat([data.drop(columns=features), normalized_features], axis=1)

# Ensure 'distance_group', 'shout_level', and 'affect' columns are preserved
if 'distance_group' not in normalized_data.columns:
    normalized_data['distance_group'] = data['distance_group']
if 'shout_level' not in normalized_data.columns:
    normalized_data['shout_level'] = data['shout_level']
normalized_data[category_column] = data[category_column]

# General radar plot function
def create_radar_plot(data, categories, title, filename, group_col, path, y_min, y_max):
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # Sorting the groups to control legend order
    sorted_data = data.sort_values(by=group_col)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(range(len(sorted_data)))

    for i, row in sorted_data.iterrows():
        values = row.drop(group_col).values.flatten().tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[i], label=row[group_col])
        ax.plot(angles, values, linewidth=2, color=colors[i])

    # Number the axes
    axis_labels = list(range(1, len(categories) + 1))
    plt.xticks(angles[:-1], axis_labels, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(y_min, y_max)  # Adjust based on the data
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title=group_col)

    # Legend for features
    legend_elements = [f"{i}: {category}" for i, category in enumerate(categories, 1)]
    legend_text = "\n".join(legend_elements)
    plt.gcf().text(0.8, 0.2, legend_text, fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.close()

# Generate radar plots for different groupings

# Plot 1: By distance groups
distance_group_data = normalized_data.groupby('distance_group')[features].mean().reset_index()
create_radar_plot(
    distance_group_data,
    features,
    "Features by Phone Position",
    "radar_plot_distance_all.png",
    group_col='distance_group',
    path=same_scale_results_folder,
    y_min=-0.4,
    y_max=0.5
)
create_radar_plot(
    distance_group_data,
    features,
    "Features by Phone Position",
    "radar_plot_distance_all.png",
    group_col='distance_group',
    path=diff_scale_results_folder,
    y_min=-0.1,
    y_max=0.1
)

# Plot 2: By shout levels
shout_level_data = normalized_data.groupby('shout_level')[features].mean().reset_index()
create_radar_plot(
    shout_level_data,
    features,
    "Features by Shout Levels",
    "radar_plot_shout_levels.png",
    group_col='shout_level',
    path=same_scale_results_folder,
    y_min=-0.4,
    y_max=0.5
)
create_radar_plot(
    shout_level_data,
    features,
    "Features by Shout Levels",
    "radar_plot_shout_levels.png",
    group_col='shout_level',
    path=diff_scale_results_folder,
    y_min=-0.1,
    y_max=0.1
)

# Plot 3: For each emotion by shout levels
for emotion in normalized_data[category_column].unique():
    emotion_data = normalized_data[normalized_data[category_column] == emotion]
    if not emotion_data.empty:
        emotion_shout_level_data = emotion_data.groupby('shout_level')[features].mean().reset_index()
        create_radar_plot(
            emotion_shout_level_data,
            features,
            f"Features by Shout Levels - {emotion}",
            f"radar_plot_shout_levels_{emotion.lower().replace(' ', '_')}.png",
            group_col='shout_level',
            path=same_scale_results_folder,
            y_min=-0.4,
            y_max=0.5
        )
        create_radar_plot(
            emotion_shout_level_data,
            features,
            f"Features by Shout Levels - {emotion}",
            f"radar_plot_shout_levels_{emotion.lower().replace(' ', '_')}.png",
            group_col='shout_level',
            path=diff_scale_results_folder,
            y_min=-0.3,
            y_max=0.3
        )

# Plot 4: Shout data by distance group
shout_data = normalized_data[normalized_data['shout_level'] == 'shout ']
shout_distance_group_data = shout_data.groupby('distance_group')[features].mean().reset_index()
create_radar_plot(
    shout_distance_group_data,
    features,
    "Features by Phone Position - Shout",
    "radar_plot_shout_distance_groups.png",
    group_col='distance_group',
    path=same_scale_results_folder,
    y_min=-0.4,
    y_max=0.5
)
create_radar_plot(
    shout_distance_group_data,
    features,
    "Features by Phone Position - Shout",
    "radar_plot_shout_distance_groups.png",
    group_col='distance_group',
    path=diff_scale_results_folder,
    y_min=-0.2,
    y_max=0.1
)

# Plot 5: No-shout data by distance group
no_shout_data = normalized_data[normalized_data['shout_level'] == 'no-shout']
no_shout_distance_group_data = no_shout_data.groupby('distance_group')[features].mean().reset_index()
create_radar_plot(
    no_shout_distance_group_data,
    features,
    "Features by Phone Position - No Shout",
    "radar_plot_no_shout_distance_groups.png",
    group_col='distance_group',
    path=same_scale_results_folder,
    y_min=-0.4,
    y_max=0.5
)
create_radar_plot(
    no_shout_distance_group_data,
    features,
    "Features by Phone Position - No Shout",
    "radar_plot_no_shout_distance_groups.png",
    group_col='distance_group',
    path=diff_scale_results_folder,
    y_min=-0.2,
    y_max=0.1
)

# Plot 6: For each emotion by distance groups 
for emotion in normalized_data[category_column].unique():
    emotion_data = normalized_data[normalized_data[category_column] == emotion]
    if not emotion_data.empty:
        emotion_distance_group_data = emotion_data.groupby('distance_group')[features].mean().reset_index()
        create_radar_plot(
            emotion_distance_group_data,
            features,
            f"Features by Phone Position - {emotion}",
            f"radar_plot_distance_{emotion.lower().replace(' ', '_')}.png",
            group_col='distance_group',
            path=same_scale_results_folder,
            y_min=-0.4,
            y_max=0.5
        )
        create_radar_plot(
            emotion_distance_group_data,
            features,
            f"Features by Phone Position - {emotion}",
            f"radar_plot_distance_{emotion.lower().replace(' ', '_')}.png",
            group_col='distance_group',
            path=diff_scale_results_folder,
            y_min=-0.4,
            y_max=0.5
        )

# Plot 7: Low and High Arousal Emotions Features
arousal_groups = {
    "High Arousal": ["anger", "surprise", "fear", "joy"],
    "Low Arousal": ["neutral", "sadness", "disgust"]
}

for arousal_group, emotions in arousal_groups.items():
    arousal_data = normalized_data[normalized_data[category_column].isin(emotions)]
    if not arousal_data.empty:
        arousal_group_data = arousal_data.groupby(category_column)[features].mean().reset_index()
        create_radar_plot(
            arousal_group_data,
            features,
            f"Features for {arousal_group} Emotions",
            f"radar_plot_{arousal_group.lower().replace(' ', '_')}.png",
            group_col=category_column,
            path=same_scale_results_folder,
            y_min=-0.4,
            y_max=0.5
        )
        create_radar_plot(
            arousal_group_data,
            features,
            f"Features for {arousal_group} Emotions",
            f"radar_plot_{arousal_group.lower().replace(' ', '_')}.png",
            group_col=category_column,
            path=diff_scale_results_folder,
            y_min=-0.2,
            y_max=0.2
        )