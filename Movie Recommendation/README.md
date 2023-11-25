# Movie Recommendation System using Surprise Library

This project involves building a movie recommendation system using the Surprise library. It uses the SVD algorithm to perform collaborative filtering for movie recommendations.

## Overview

This Python script creates a recommendation system based on user ratings for movies. It utilizes the Surprise library, which specializes in building and analyzing recommender systems.

## Getting Started

### Prerequisites

Ensure you have the necessary libraries installed:

- pandas
- Surprise
- tqdm

### Installation

Clone the repository:

```bash
git clone https://github.com/your_username/movie-recommendation.git
cd movie-recommendation
```

#### Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Code

Ensure you have your dataset files 'movies.csv' and 'ratings.csv' in the project directory.

Run the `movie_recommendation.py` script:

```bash
python movie_recommendation.py
```

This script builds a movie recommendation system using the Surprise library.

## Steps in the Script

1. **Loading Imports:** Loading necessary libraries and dependencies.
2. **Loading Data from CSV:** Reading movies and ratings data from CSV files.
3. **Merging Datasets:** Merging movies and ratings datasets.
4. **Load Surprise Dataset:** Transforming the dataset to Surprise format.
5. **Train-Test Split:** Splitting the dataset into training and testing sets.
6. **Building Model (SVD):** Building a collaborative filtering model using SVD.
7. **Training the Model:** Training the collaborative filtering model.
8. **Saving the Model:** Saving the trained model and its weights.

## Models and Files

- `movie_recommendation_model.pkl`: Saved Surprise collaborative filtering model.

## Acknowledgments

- The project uses the Surprise library for collaborative filtering.
