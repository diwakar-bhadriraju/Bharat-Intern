# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tqdm import tqdm
from joblib import dump, load

# Load the dataset
df = pd.read_csv('Housing.csv')

# Separate features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])
    ],
    remainder='passthrough'
)

# Train a scikit-learn linear regression model
linear_model = LinearRegression()
linear_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', linear_model)
])
linear_pipeline.fit(X_train, y_train)

# Save the scikit-learn linear regression model
dump(linear_pipeline, 'linear_model.joblib')

# Convert categorical columns to numerical for neural network
X_train_nn = preprocessor.transform(X_train)
X_test_nn = preprocessor.transform(X_test)

# Train a neural network model using TensorFlow
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model with tqdm for progress bar
epochs = 50
batch_size = 32
verbose = 2
for epoch in tqdm(range(epochs), desc='Neural Network Training'):
    model.fit(X_train_nn, y_train, epochs=1, batch_size=batch_size, verbose=verbose)

# Save the entire TensorFlow neural network model in the native Keras format
model.save('neural_network_model')

# Save only the weights of the neural network model
model.save_weights('neural_network_weights.h5')

# Make predictions on the test set using the neural network model
nn_predictions = model.predict(X_test_nn).flatten()

# Evaluate the neural network model
nn_rmse = mean_squared_error(y_test, nn_predictions, squared=False)
print(f'Neural Network RMSE: {nn_rmse}')
