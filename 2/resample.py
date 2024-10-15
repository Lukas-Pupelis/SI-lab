import pandas as pd
from sklearn.utils import resample

# Load the dataset from a CSV file (make sure to replace 'your_file.csv' with the actual file path)
iris_data = pd.read_csv('your_file.csv', delimiter=';')

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
augmented_data.to_csv('augmented_iris_data.csv', index=False, header=False)