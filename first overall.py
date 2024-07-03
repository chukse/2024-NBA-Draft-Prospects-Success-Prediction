import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('nba_draft_prospects.csv')

# Select the features and target variable
features = ['PTS', 'AST', 'DREB', 'STL', 'BLK', 'Avg. Rank', 'Ovr. Rank', 'Lane (sec)', '2PA/36', 'MP', 'FT%', 'GP', 'Age', '2P%']
target = 'Rank'

# Function to clean data
def clean_data(value):
    if isinstance(value, str):
        if value in ['#DIV/0!', '-']:
            return np.nan
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

# Clean the data
for column in features + [target]:
    data[column] = data[column].apply(clean_data)
data = data.dropna(subset=[target])
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with imputer, scaler, and HistGradientBoostingRegressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict the draft rank for all prospects
def predict_draft_ranks(prospects):
    return pipeline.predict(prospects)

# Predict draft ranks for all prospects
all_predictions = predict_draft_ranks(X)

# Add predictions to the original dataframe
data['predicted_rank'] = all_predictions

# Sort the dataframe by predicted rank
data_sorted = data.sort_values('predicted_rank')

# Display the top 5 predicted picks
print("\nTop 5 Predicted Draft Picks:")
for i, (index, row) in enumerate(data_sorted.head().iterrows(), 1):
    print(f"{i}. {row['Player']} (Predicted Rank: {row['predicted_rank']:.2f})")
