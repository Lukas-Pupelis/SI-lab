import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (original or augmented)
iris_data = pd.read_csv('your_file.csv', delimiter=';')

# Give the columns meaningful names (if not already done)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Create a scatter plot for 'sepal_length' vs 'sepal_width'
plt.figure(figsize=(10, 6))

# Use seaborn to create the scatter plot, coloring by species
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_data, palette='Set1')

# Add titles and labels
plt.title('Sepal Length vs Sepal Width (by Species)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Display the plot
plt.show()