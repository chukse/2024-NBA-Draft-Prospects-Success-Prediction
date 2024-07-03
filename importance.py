import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re

# Load the dataset
file_path = 'C:/Users/Chuks/Documents/nba draft/nba_draft_prospects.csv'
data = pd.read_csv(file_path)

# Function to convert height from format 6'4.5" to inches
def height_to_inches(height):
    if isinstance(height, str):
        match = re.match(r"(\d+)'(\d+(\.\d+)?)\"", height)
        if match:
            feet = int(match.group(1))
            inches = float(match.group(2))
            return feet * 12 + inches
    return None

# Apply the height conversion
data['Height'] = data['Height'].apply(height_to_inches)

# Drop non-numeric columns that won't be used as features
data = data.drop(columns=['Player', 'Position', 'School', 'Birthday', 'Info'])

# Convert all columns to numeric, coercing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')
#print(data.isnull().sum())
#print(f"Original shape: {data.shape}")

# Drop rows with missing values
data = data.dropna(subset=['Rank', 'Height'])

# Select the features and target variable
X = data.drop(columns=['Rank'])
y = data['Rank']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the most important features
print(importance_df)

