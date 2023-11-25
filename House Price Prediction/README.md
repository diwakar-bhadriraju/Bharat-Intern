# House Price Prediction

This project involves predicting house prices using both linear regression from scikit-learn and a neural network implemented using TensorFlow/Keras.

## Getting Started

### Prerequisites

Make sure you have the necessary libraries installed:

- pandas
- scikit-learn
- TensorFlow
- tqdm
- joblib

### Installation

Clone the repository:

```bash
git clone https://github.com/your_username/house-price-prediction.git
cd house-price-prediction
```

#### Install the required libraries:

```bash
pip install -r requirements.txt
```

### Usage

#### Ensure you have your dataset named 'Housing.csv' in the project directory.

Run the `house_price_prediction.py` script:

```bash
python house_price_prediction.py
```

This script trains a linear regression model and a neural network for house price prediction.

### Files

- `house_price_prediction.py`: Main Python script for house price prediction.
- `Housing.csv`: Dataset containing house-related features and prices.

## Results

After running the script, the model will be trained, and the RMSE (Root Mean Squared Error) for the neural network will be displayed.

## Models and Files

- `linear_model.joblib`: Saved scikit-learn linear regression model.
- `neural_network_model`: Entire TensorFlow neural network model in Keras format.
- `neural_network_weights.h5`: Saved weights of the neural network model.

## Acknowledgments

- The project utilizes scikit-learn, TensorFlow, and other libraries for machine learning.
