import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from tqdm import tqdm
import joblib

# Step 1: Loading imports
print("Step 1: Loading imports")
try:
    import pandas as pd
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    from tqdm import tqdm
    import joblib
    print("Imports loaded successfully\n")
except Exception as e:
    print(f"Error loading imports: {e}\n")
    exit()

# Step 2: Loading data from CSV
print("Step 2: Loading data from CSV")
try:
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    print("Data loaded successfully\n")
except Exception as e:
    print(f"Error loading data: {e}\n")
    exit()

# Step 3: Merging datasets
print("Step 3: Merging datasets")
try:
    data = pd.merge(ratings, movies, on='movieId')
    print("Datasets merged successfully\n")
except Exception as e:
    print(f"Error merging datasets: {e}\n")
    exit()

# Step 4: Load Surprise dataset
print("Step 4: Load Surprise dataset")
try:
    reader = Reader(rating_scale=(0.5, 5))
    surprise_data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    print("Surprise dataset loaded successfully\n")
except Exception as e:
    print(f"Error loading Surprise dataset: {e}\n")
    exit()

# Step 5: Train-test split
print("Step 5: Train-test split")
try:
    trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)
    print("Train-test split completed successfully\n")
except Exception as e:
    print(f"Error in train-test split: {e}\n")
    exit()

# Step 6: Build collaborative filtering model (SVD)
print("Step 6: Build model (SVD)")
try:
    model = SVD()
    print("Model built successfully\n")
except Exception as e:
    print(f"Error building model: {e}\n")
    exit()

# Step 7: Training the model
print("Step 7: Training the model")
try:
    tqdm.pandas()  # Enable progress bar for pandas operations
    model.fit(trainset)
    print("Training completed successfully\n")
except Exception as e:
    print(f"Error during training: {e}\n")
    exit()

# Step 8: Save the model, its weights, and checkpoints
print("Step 8: Save the model, its weights, and checkpoints")
try:
    joblib.dump(model, 'movie_recommendation_model.pkl')
    print("Model saved successfully\n")
except Exception as e:
    print(f"Error saving model: {e}\n")
    exit()

# Final Step: Completion
print("Recommendation system built successfully!")
