import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C:/Users/37067/Desktop/skaitmeninis intelektas/SI-lab/2')

# Load the dataset (original or augmented)
orig_iris_data = pd.read_csv('iris-data.csv', delimiter=';')

aug_iris_data = pd.read_csv('formatted_augmented_iris_data.csv', delimiter=';')

# Give the columns meaningful names (if not already done)
orig_iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

aug_iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Create a scatter plot for 'sepal_length' vs 'sepal_width'
plt.figure(figsize=(10, 6))

# Use seaborn to create the scatter plot, coloring by species
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=orig_iris_data, palette='Set1')

# Add titles and labels
plt.title('Sepal Length vs Sepal Width (by Species)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Display the plot
plt.show()

# Use seaborn to create the scatter plot, coloring by species
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=aug_iris_data, palette='Set2')

# Add titles and labels
plt.title('Sepal Length vs Sepal Width (by Species)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Display the plot
plt.show()