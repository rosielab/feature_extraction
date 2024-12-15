import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data_path = "./updated_csv2.csv"  # Replace with your CSV file path
df = pd.read_csv(data_path)

# Function to normalize each value in the column independently
def normalize_column(cell, col):
    return (cell - col.min()) / (col.max() - col.min()) if col.max() != col.min() else cell

# Normalize relevant features
feature_columns = [
    'F1frequency_range', 'slope0-500_mean', 'slope500-1500_mean', 'F1frequency_mean',
    'shimmer_mean', 'jitter_mean', 'loudness_mean'
]

# Apply normalization for each feature column
for feature in feature_columns:
    df[feature] = df.apply(lambda row: normalize_column(row[feature], df[feature]), axis=1)

# Define emotion groups
emotions = ['anger', 'joy', 'sadness', 'surprise', 'neutral', 'fear', 'disgust']

# Feature-to-number mapping for legend
feature_labels = {
    1: 'F1 Frequency Range',
    2: 'Slope 0-500 Hz Mean',
    3: 'Slope 500-1500 Hz Mean',
    4: 'F1 Frequency Mean',
    5: 'Shimmer Mean',
    6: 'Jitter Mean',
    7: 'Loudness Mean'
}

# Create the output folder
output_folder = "shout_emotion_plots"
os.makedirs(output_folder, exist_ok=True)

# Radar plot function
def plot_radar(data, title, file_name):
    categories = list(range(1, len(feature_columns) + 1))  # Use numbers 1–7
    num_vars = len(categories)

    # Create a radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for shout_value, color in zip([1, 0], ['red', 'blue']):
        subset = data[data['shout'] == shout_value]
        if not subset.empty:
            values = subset[feature_columns].mean().tolist()
            values += values[:1]  # Close the circle
            ax.plot(angles, values, color=color, linewidth=2, label=f"Shout = {shout_value}")
            ax.fill(angles, values, color=color, alpha=0.25)

    # Add numbers (1–7) around the plot
    ax.set_yticks([])  # Hide radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, rotation=45)

    # Add shout legend
    shout_legend = ax.legend(loc='upper right', title="Shout Status", bbox_to_anchor=(1.1, 1))

    # Add feature legend (box, positioned outside the plot to avoid overlap)
    from matplotlib.lines import Line2D
    feature_handles = [Line2D([0], [0], color='none', label=f"{key}: {value}") for key, value in feature_labels.items()]
    feature_legend = plt.legend(
        handles=feature_handles,
        loc='center left',  # Position to the left of the plot
        bbox_to_anchor=(0.9, 0),  # Offset legend (x, y) outside plot area
        title="Features",
        fontsize='small',
        frameon=True  # Add a box around the legend
    )
    fig.add_artist(shout_legend)  # Ensure the shout legend remains
    fig.add_artist(feature_legend)  # Add the feature legend

    ax.set_title(title, size=14, pad=20)
    plt.tight_layout()

    # Save the plot as a PNG file
    file_path = os.path.join(output_folder, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")  # Ensure all elements fit in the saved image
    plt.close()

# Generate and save radar plots for each emotion
for emotion in emotions:
    emotion_data = df[df['affect'] == emotion]
    plot_title = f"Radar Plot for {emotion.capitalize()} Emotion"
    file_name = f"{emotion}_radar_plot.png"
    plot_radar(emotion_data, title=plot_title, file_name=file_name)
