# Scaling values below for easier reading
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# Load the CSV file
DATA_CSV_FILE = './processed_data_with_features_2.csv'  # Replace with your actual path
data = pd.read_csv(DATA_CSV_FILE)

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
data = data[data[category_column] != 'saddness']

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

# Separate data based on the groups
group_1_data = normalized_data[normalized_data[category_column].isin(group_1_acts)]
group_2_data = normalized_data[normalized_data[category_column].isin(group_2_acts)]

# Prepare data for radar plot
categories = features  # Features
num_vars = len(categories)

def create_radar_plot(data, categories, title):
    # Initialize figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Adjust the color palette dynamically based on the number of data points
    num_data_points = len(data)
    colors = plt.cm.tab10(range(min(num_data_points, 10)))  # Limit colors to 10
    
    # Iterate through each category (affect)
    for i, row in data.iterrows():
        # Convert data to angle format
        values = row.drop(category_column).values.flatten().tolist()
        values += values[:1]  # Repeat the first value to close the circle
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        if row[category_column] == 'joy':
            color = 'green'
        else:
            color = colors[i % len(colors)]  # Dynamically assign color

        # Plot data
        ax.fill(angles, values, alpha=0.25, color=color, label=row[category_column])
        ax.plot(angles, values, linewidth=2, color=color)

    # Add category labels
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.yaxis.grid(True)
    ax.set_ylim(0, 1)  # Set the range of the radar plot to [0, 1]

    # Add title and legend
    plt.title(title, size=16, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()

# Plot for group 1
create_radar_plot(group_1_data, categories, "Radar Plot of Acoustic Features (Group 1: Fear, Disgust, Anger)")

# Plot for group 2
create_radar_plot(group_2_data, categories, "Radar Plot of Acoustic Features (Group 2: Neutral, Joy, Surprise)")
