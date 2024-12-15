import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_path = "./updated_toolbox.csv"  # Replace with your CSV file path
df = pd.read_csv(data_path)

# Function to normalize column
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else col

# Normalize relevant features
feature_columns = ['mean pitch', 'pitch range', 'speech rate', 'energy']
df[feature_columns] = df[feature_columns].apply(normalize_column)

# Define emotion groups
high_arousal = ['joy', 'surprise', 'anger', 'fear']
low_arousal = ['neutral', 'sadness', 'disgust']

# Define color mapping for emotions
color_map = {
    'joy': 'red',
    'surprise': 'blue',
    'anger': 'orange',
    'fear': 'purple',
    'neutral': 'green',
    'sadness': 'cyan',
    'disgust': 'magenta'
}

# T-configuration scatter plot function
def plot_t_scatter(data, emotions, title):
    plt.figure(figsize=(12, 12))

    # Horizontal scatter plot (mean pitch vs pitch range)
    plt.subplot(2, 1, 1)
    for emotion in emotions:
        subset = data[data['emotion'] == emotion]
        plt.scatter(
            subset['mean pitch'],
            subset['pitch range'],
            c=color_map[emotion],
            label=emotion,
            marker='s' if subset['shout'].iloc[0] == 1 else 'o',
            alpha=0.7
        )
    plt.title(f"{title}: Mean Pitch vs Pitch Range")
    plt.xlabel("Mean Pitch")
    plt.ylabel("Pitch Range")
    plt.legend()
    plt.grid(alpha=0.5)

    # Vertical scatter plot (speech rate vs energy)
    plt.subplot(2, 1, 2)
    for emotion in emotions:
        subset = data[data['emotion'] == emotion]
        plt.scatter(
            subset['speech rate'],
            subset['energy'],
            c=color_map[emotion],
            label=emotion,
            marker='s' if subset['shout'].iloc[0] == 1 else 'o',
            alpha=0.7
        )
    plt.title(f"{title}: Speech Rate vs Energy")
    plt.xlabel("Speech Rate")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()

# Filter data for each arousal group
high_arousal_data = df[df['emotion'].isin(high_arousal)]
low_arousal_data = df[df['emotion'].isin(low_arousal)]

# Create T-configuration scatter plots
plot_t_scatter(high_arousal_data, high_arousal, "High Arousal Emotions")
plot_t_scatter(low_arousal_data, low_arousal, "Low Arousal Emotions")
