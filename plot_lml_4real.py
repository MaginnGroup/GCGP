import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_plot_settings

global_plot_settings.set_plot_settings()

# Read the CSV file containing the LML values
df = pd.read_csv('lml_values.csv')

# Create a bar chart for each thermophysical property
for property in df['Property'].unique():
    # Filter the DataFrame for the current property
    property_df = df[df['Property'] == property]
    
    # Create labels by concatenating Model, Kernel, and Flag as strings
    labels = property_df['Model'].astype(str) + '_' + property_df['Kernel'] + '_' + property_df['Flag'].astype(str)
    
    # Find the index of the maximum LML value
    max_index = (-property_df['LML']).idxmax()
    
    # Plot the bar chart
    plt.figure(figsize=(20, 6))
    bars = plt.bar(labels, -property_df['LML'])
    
    # Highlight the bar with the maximum LML value
    max_label = f"{property_df.loc[max_index, 'Model']}_{property_df.loc[max_index, 'Kernel']}_{property_df.loc[max_index, 'Flag']}"
    bars[labels.tolist().index(max_label)].set_color('red')
    
    plt.xlabel(property)
    plt.ylabel('Log Evidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig(f'lml_bar_chart_{property}.png')
    plt.close()

print("Bar charts saved for each thermophysical property.")
