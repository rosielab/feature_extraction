import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('./updated_toolbox.csv')

# Define feature columns for the radar plot
feature_columns = ['mean pitch', 'pitch range', 'speech rate', 'energy', 
                   'shimmer', 'jitter', 'spectral slope', 'mean intensity', 'max intensity']

# Normalize features to make the comparisons meaningful
def normalize_column(col):
    if col.max() - col.min() == 0:
        return col  # Skip normalization for constant columns
    return (col - col.min()) / (col.max() - col.min())

for col in feature_columns:
    df[col] = normalize_column(df[col])

# Define high and low arousal emotions
high_arousal_emotions = ['anger', 'fear', 'joy', 'surprise']
low_arousal_emotions = ['neutral', 'sad', 'disgust']

# Filter for high and low arousal subsets
high_arousal = df[df['emotion'].isin(high_arousal_emotions)]
low_arousal = df[df['emotion'].isin(low_arousal_emotions)]

# Compute mean feature values for shout and no shout
shout_high = high_arousal[high_arousal['shout'] == 1][feature_columns].mean()
no_shout_high = high_arousal[high_arousal['shout'] == 0][feature_columns].mean()
shout_low = low_arousal[low_arousal['shout'] == 1][feature_columns].mean()
no_shout_low = low_arousal[low_arousal['shout'] == 0][feature_columns].mean()

# Radar plot function
def plot_radar(shout_values, no_shout_values, title):
    categories = feature_columns
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Prepare the data
    shout_values = np.concatenate((shout_values, [shout_values[0]]))
    no_shout_values = np.concatenate((no_shout_values, [no_shout_values[0]]))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, shout_values, label='Shout', color='red')
    ax.fill(angles, shout_values, alpha=0.25, color='red')
    ax.plot(angles, no_shout_values, label='No Shout', color='blue')
    ax.fill(angles, no_shout_values, alpha=0.25, color='blue')

    # Labels and Title
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.show()

# Plot for High Arousal
plot_radar(shout_high.values, no_shout_high.values, "High Arousal: Shout vs No Shout")

# Plot for Low Arousal
plot_radar(shout_low.values, no_shout_low.values, "Low Arousal: Shout vs No Shout")

