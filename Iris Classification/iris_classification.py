import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# Load the Iris dataset
data = pd.read_csv('iris_dataset.csv')

# Function to add random perturbation to the data
def perturb_data(data, perturbation_factor=0.05, desired_rows=1000):
    current_rows = len(data)
    rows_to_add = desired_rows - current_rows

    # Replicate existing data to match the desired number of rows
    replicated_data = pd.concat([data] * (rows_to_add // current_rows + 1), ignore_index=True)
    
    # Perturb the replicated data by adding random noise to numerical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    perturbed_data = replicated_data.copy()

    for column in numerical_columns:
        perturbed_data[column] += np.random.normal(0, perturbation_factor, size=len(perturbed_data))

    # Combine original data and perturbed data
    augmented_data = pd.concat([data, perturbed_data.iloc[:rows_to_add, :]], ignore_index=True)
    
    return augmented_data

# Augment the dataset using random perturbation
augmented_data = perturb_data(data, perturbation_factor=0.05, desired_rows=1000)

# Save the augmented dataset
augmented_data.to_csv('augmented_iris_dataset.csv', index=False)


# Load the Iris dataset
data = pd.read_csv('augmented_iris_dataset.csv')

# Data preprocessing
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# One-hot encoding the target variable (Species)
y_encoded = pd.get_dummies(y)

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2)

# Define and train the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_encoded, epochs=100, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)

# Save the trained model
model.save('iris_classifier_neural_network.h5')

# Make predictions using the saved model
sepal_length_cm = float(input('Enter sepal length (cm): '))
sepal_width_cm = float(input('Enter sepal width (cm): '))
petal_length_cm = float(input('Enter petal length (cm): '))
petal_width_cm = float(input('Enter petal width (cm): '))

new_data = [[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]]

# Load the saved model
loaded_model = tf.keras.models.load_model('iris_classifier_neural_network.h5')

# Make predictions
predictions = loaded_model.predict(new_data)
predicted_species_index = tf.argmax(predictions, axis=1)

# Convert the predicted species index into the corresponding species name
species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
predicted_species_name = species_names[predicted_species_index[0]]

print('Predicted species:', predicted_species_name)