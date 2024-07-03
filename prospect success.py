import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectFromModel
from fuzzywuzzy import process
import time

# Load the datasets
start_time = time.time()
historical_data = pd.read_csv('C:/Users/Chuks/Documents/nba draft/draft-data-20-years.csv')
prospects_data = pd.read_csv('C:/Users/Chuks/Documents/nba draft/nba_draft_prospects.csv')
nba2024_draft_results = pd.read_csv('C:/Users/Chuks/Documents/nba draft/2024_NBA_Draft_results.csv')
college_ball_data = pd.read_csv('C:/Users/Chuks/Documents/nba draft/CollegeBasketballPlayers2009-2021_full.csv')
nba_raptor_stats = pd.read_csv('C:/Users/Chuks/Documents/nba draft/av_modern_RAPTOR_by_player.csv')
print("Data loading time:", time.time() - start_time)

# Clean the data
start_time = time.time()
def clean_data(value):
    if isinstance(value, str):
        if value.startswith('http'):  # If it's a URL
            return value  # Keep URLs as is
        if value in ['#DIV/0!', '-']:
            return np.nan
        try:
            return float(value)
        except ValueError:
            return value
    return value

def apply_clean_data(df):
    for column in df.columns:
        df[column] = df[column].apply(clean_data)
    return df

historical_data = apply_clean_data(historical_data)
prospects_data = apply_clean_data(prospects_data)
college_ball_data = apply_clean_data(college_ball_data)
nba_raptor_stats = apply_clean_data(nba_raptor_stats)
print("Data cleaning time:", time.time() - start_time)

# Normalize player names
start_time = time.time()
def normalize_name(name):
    parts = name.split()
    parts.sort()
    return ' '.join(parts)

historical_data['normalized_name'] = historical_data['Player'].apply(normalize_name)
nba_raptor_stats['normalized_name'] = nba_raptor_stats['player_name'].apply(normalize_name)
print("Normalization time:", time.time() - start_time)

# Fuzzy match and merge the dataframes
start_time = time.time()
def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=2):
    s = df_2[key2].tolist()
    m = df_1[key1].apply(lambda x: process.extractOne(x, s, score_cutoff=threshold))
    df_1['match'] = m
    df_1['match_name'] = df_1['match'].apply(lambda x: x[0] if x else np.nan)
    df_1['match_score'] = df_1['match'].apply(lambda x: x[1] if x else np.nan)
    df_1 = df_1.drop(columns=['match'])
    df_merged = pd.merge(df_1, df_2, left_on='match_name', right_on=key2, how='left')
    return df_merged

merged_historical_raptor = fuzzy_merge(historical_data, nba_raptor_stats, 'normalized_name', 'normalized_name', threshold=80)
print("Fuzzy merging time:", time.time() - start_time)

# Model Training and Hyperparameter Tuning
start_time = time.time()
# Identify numeric columns
numeric_columns = merged_historical_raptor.select_dtypes(include=[np.number]).columns
historical_data_numeric = merged_historical_raptor[numeric_columns]

# List potential targets
potential_targets = ['WS', 'VORP', 'PER', 'raptor_onoff_offense', 'raptor_onoff_defense',
                     'raptor_offense','raptor_defense','war_total']

# Check data availability
for target in potential_targets:
    if target in merged_historical_raptor.columns:
        non_null_count = merged_historical_raptor[target].count()
        total_count = len(merged_historical_raptor)
        print(f"{target}: {non_null_count}/{total_count} non-null values")

# Correlation with other metrics
for target in potential_targets:
    if target in historical_data_numeric.columns:
        correlations = historical_data_numeric.corr()[target].sort_values(ascending=False)
        print(f"\nTop correlations for {target}:")
        print(correlations.head())

# Choose targets
targets = ['WS/48', 'PPG', 'VORP', 'raptor_onoff_offense', 'raptor_onoff_defense',
                     'raptor_offense','raptor_defense','war_total','predator_defense','predator_total']

# Drop rows with missing target values in historical data
merged_historical_raptor = merged_historical_raptor.dropna(subset=targets)

# Define features and ensure they exist in both datasets
available_historical_features = merged_historical_raptor.columns.tolist()
available_prospects_features = prospects_data.select_dtypes(include=[np.number]).columns.tolist()

# Identify common features
common_features = list(set(available_historical_features).intersection(available_prospects_features))

# Filter out the targets and any non-relevant columns
common_features = [feature for feature in common_features if feature not in ['Player', 'Pk'] + targets]

print("\nCommon Features Used for Prediction:")
print(common_features)

# Add error handling for missing columns
missing_features = set(common_features) - set(prospects_data.columns)
if missing_features:
    print(f"Warning: The following features are missing from the prospects data: {missing_features}")
    common_features = [f for f in common_features if f not in missing_features]

# Create feature matrix X and target matrix Y
X = merged_historical_raptor[common_features]
Y = merged_historical_raptor[targets]

# Feature selection
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
selector.fit(X, Y)
X_selected = selector.transform(X)
selected_features = [feature for feature, selected in zip(common_features, selector.get_support()) if selected]

print("\nSelected Features:")
print(selected_features)

# Use selected features
X = pd.DataFrame(X_selected, columns=selected_features)

# Split historical data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a pipeline with imputer, scaler, and MultiOutputRegressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X, Y, cv=5, scoring='r2')
print(f"Cross-validation R2 scores: {cv_scores}")
print(f"Mean R2 score: {cv_scores.mean()}")

# Hyperparameter tuning
param_grid = {
    'regressor__estimator__n_estimators': [100, 200, 300],
    'regressor__estimator__max_depth': [10, 20, 30],
    'regressor__estimator__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X, Y)

print("Best parameters:", grid_search.best_params_)

# Use the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
Y_pred = best_model.predict(X_test)

# Feature importance
importances = best_model.named_steps['regressor'].estimators_[0].feature_importances_
feature_importance = pd.DataFrame({'feature': selected_features, 'importance': importances})
print(feature_importance.sort_values('importance', ascending=False))

# Evaluate the model
for i, target in enumerate(targets):
    mse = mean_squared_error(Y_test[target], Y_pred[:, i])
    r2 = r2_score(Y_test[target], Y_pred[:, i])
    print(f'{target} - Mean Squared Error: {mse}, R2 Score: {r2}')

# Function to predict the success of current draft prospects
def predict_success(prospects):
    return best_model.predict(prospects)

# Prepare the prospects data
X_prospects = prospects_data[selected_features]

# Predict success for all prospects
success_predictions = predict_success(X_prospects)

# Add predictions to the original dataframe
for i, target in enumerate(targets):
    prospects_data[f'Predicted_{target}'] = success_predictions[:, i]

# Sort the dataframe by all predicted markers
sort_columns = [f'Predicted_{target}' for target in targets]
sorted_prospects_data = prospects_data.sort_values(by=sort_columns, ascending=False)

# Export the sorted dataframe to a CSV file
sorted_prospects_data.to_csv('sorted_prospects_data.csv', index=False)
print("Data sorted and exported to sorted_prospects_data.csv")

# Fuzzy match and merge the dataframes
merged_df = fuzzy_merge(sorted_prospects_data, nba2024_draft_results, 'Player', 'Player', threshold=80)

# Sort the merged dataframe by predicted markers
merged_df = merged_df.sort_values(by=sort_columns, ascending=False)

# Display the merged dataframe
print("\nMerged DataFrame:")
print(merged_df.head())

# Save merged dataframe to a CSV file
merged_df.to_csv('merged_prospects_draft_results.csv', index=False)
print("Merged dataframe saved to merged_prospects_draft_results.csv")






