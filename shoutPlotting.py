import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data_path = "./updated_toolbox.csv"  # Replace with your CSV file path
df = pd.read_csv(data_path)

# Function to normalize column
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else col

# Normalize relevant features
feature_columns = [
    'mean pitch', 'pitch range', 'speech rate', 'energy', 
    'shimmer', 'jitter', 'spectral slope', 'mean intensity', 'max intensity'
]
df[feature_columns] = df[feature_columns].apply(normalize_column)

# Define emotion groups
emotions = ['anger', 'joy', 'sadness', 'surprise', 'neutral', 'fear', 'disgust']

# Radar plot function
def plot_radar(data, title):
    categories = feature_columns
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

    # Add feature names around the plot
    ax.set_yticks([])  # Hide radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, rotation=45)

    # Add legend for shout values
    ax.legend(loc='upper right', title="Shout Status")

    ax.set_title(title, size=14, pad=20)
    plt.tight_layout()
    plt.show()

# Generate radar plots for each emotion
for emotion in emotions:
    emotion_data = df[df['emotion'] == emotion]
    plot_radar(emotion_data, title=f"Radar Plot for {emotion.capitalize()} Emotion")
