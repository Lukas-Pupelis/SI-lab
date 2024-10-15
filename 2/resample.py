import pandas as pd
from sklearn.utils import resample
import os
os.chdir('C:/Users/37067/Desktop/skaitmeninis intelektas/SI-lab/2')

iris_data = pd.read_csv('iris-data.csv', delimiter=';')

# Separate the classes based on the last column (assuming the last column contains the class labels)
class_0 = iris_data[iris_data[iris_data.columns[-1]] == 0]  # Class 0
class_1 = iris_data[iris_data[iris_data.columns[-1]] == 1]  # Class 1

# Resample each class to 200 samples
class_0_upsampled = resample(class_0, replace=True, n_samples=200, random_state=42)
class_1_upsampled = resample(class_1, replace=True, n_samples=200, random_state=42)

# Combine the upsampled datasets
augmented_data = pd.concat([class_0_upsampled, class_1_upsampled])

# Shuffle the data to mix both classes
augmented_data = augmented_data.sample(frac=1).reset_index(drop=True)

# Save the augmented data to a new CSV file
augmented_data.to_csv('augmented-iris-data.csv', index=False, header=False)