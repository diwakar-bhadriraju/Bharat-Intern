# Iris Classification with Neural Networks

This project utilizes neural networks to classify iris species based on provided features. The neural network model is implemented using TensorFlow's Keras API.

## Overview

The project involves the following steps:

### Data Augmentation

The dataset is initially loaded and then augmented using random perturbation to expand it to the desired number of rows.

### Data Preprocessing

- The augmented dataset is preprocessed by extracting features (Sepal Length, Sepal Width, Petal Length, and Petal Width) and the target variable (Species).
- One-hot encoding is applied to the target variable (Species) to convert categorical values into numerical form.

### Model Building and Training

- A Sequential neural network model is constructed using Keras with three dense layers.
- The model is compiled using 'categorical_crossentropy' as the loss function and 'adam' as the optimizer.
- It is then trained using the training dataset for 100 epochs with a batch size of 32.

### Model Evaluation and Prediction

- The trained model is evaluated on the test set to measure its accuracy.
- Finally, the model is used to predict the iris species based on user input for sepal and petal dimensions.

## Getting Started

Follow these steps to get started with the Iris Classification project:

### Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/diwakar-bhadriraju/Bharat-Intern.git
   cd iris-classification-neural-network
   ```

#### Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Load and preprocess the dataset.
2. Run the `iris_classification.py` script to train the neural network model and save it.
3. Use the saved model to predict the iris species based on provided sepal and petal dimensions.

### Files

- `iris_dataset.csv`: Original dataset.
- `augmented_iris_dataset.csv`: Augmented dataset.
- `iris_classifier_neural_network.h5`: Saved neural network model.

Feel free to experiment with different neural network architectures or hyperparameters for potential performance improvements.

## Contributions

Contributions to enhance the model, add features, or fix issues are welcome. Please open an issue or create a pull request for discussion.
