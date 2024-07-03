import pandas as pd


import pandas as pd

# Load the dataset
file_path = 'C:/Users/Chuks/Documents/nba draft/modern_RAPTOR_by_player.csv'
df = pd.read_csv(file_path)

# Function to check if a string has a "'" prefix and convert it to a float
def convert_to_float(entry):
    if isinstance(entry, str) and entry.startswith("'"):
        try:
            return float(entry[1:])
        except ValueError:
            return entry
    return entry

# Apply the conversion function to the entire DataFrame
df = df.applymap(convert_to_float)

# Remove rows with any entries that have "'" prefix and couldn't be converted to float
df = df.dropna()

# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'C:/Users/Chuks/Documents/nba draft/cleaned_modern_RAPTOR_by_player.csv'
df.to_csv(cleaned_file_path, index=False)

cleaned_file_path

df = pd.read_csv(cleaned_file_path)

# Group by the player's name and average their statistics
# The player's name is in the column named 'player_name'
averaged_df = df.groupby('player_name', as_index=False).mean(numeric_only=True)
print(averaged_df)
# Save the averaged DataFrame to a new CSV file
averaged_file_path = 'C:/Users/Chuks/Documents/nba draft/av_modern_RAPTOR_by_player.csv'
averaged_df.to_csv(averaged_file_path, index=False)

averaged_file_path