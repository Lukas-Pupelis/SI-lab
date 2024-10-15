import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C:/Users/37067/Desktop/skaitmeninis intelektas/SI-lab/2')

orig_iris_data = pd.read_csv('iris-data.csv', delimiter=';')
aug_iris_data = pd.read_csv('formatted_augmented_iris_data.csv', delimiter=';')

orig_iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
aug_iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=orig_iris_data, palette='Set1', ax=ax[0])
ax[0].set_title('Original: Sepal Length vs Sepal Width (by Species)')
ax[0].set_xlabel('Sepal Length (cm)')
ax[0].set_ylabel('Sepal Width (cm)')

sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=aug_iris_data, palette='Set1', ax=ax[1])
ax[1].set_title('Augmented: Sepal Length vs Sepal Width (by Species)')
ax[1].set_xlabel('Sepal Length (cm)')
ax[1].set_ylabel('Sepal Width (cm)')

plt.tight_layout()
plt.show()