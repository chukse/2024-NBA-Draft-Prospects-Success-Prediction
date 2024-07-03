import pandas as pd

# Data to be formatted into a DataFrame
data = {
    "PK(OVR)": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    "TEAM": ["Toronto", "Utah", "Milwaukee", "Portland", "San Antonio", "Indiana", "Minnesota", "New York", "Memphis", "Portland", "Philadelphia", "Charlotte", "Miami", "Houston", "Sacramento", "LA", "Orlando", "San Antonio", "Indiana", "Indiana", "New York", "Golden State", "Detroit", "Boston", "LA Lakers", "Phoenix", "Memphis", "Dallas"],
    "NAME": ["Jonathan Mogbo", "Kyle Filipowski", "Tyler Smith", "Tyler Kolek", "Johnny Furphy", "Juan Núñez", "Bobi Klintman", "Ajay Mitchell", "Jaylen Wells", "Osasere Ighodaro", "Adem Bona", "KJ Simpson", "Nikola Djurisić", "Pelle Larsson", "Jamal Shead", "Cameron Christie", "Antonio Reeves", "Harrison Ingram", "Tristen Newton", "Enrique Freeman", "Melvin Ajinça", "Quinten Post", "Cam Spencer", "Anton Watson", "Bronny James", "Kevin McCullar Jr.", "Ulrich Chomche", "Ariel Hukporti"],
    "HT": ["6-6", "6-10", "6-9", "6-1", "6-7", "6-4", "6-8", "6-3", "6-6", "6-9", "6-8", "6-0", "6-7", "6-5", "6-0", "6-4", "6-4", "6-5", "6-3", "6-7", "6-7", "7-0", "6-3", "6-7", "6-1", "6-5", "6-10", "6-10"],
    "WT": [217, 230, 224, 197, 189, 206, 212, 197, 206, 222, 243, 187, 209, 212, 201, 190, 187, 234, 192, 212, 214, 244, 202, 233, 210, 206, 232, 246],
    "POS": ["C", "PF", "SF", "PG", "SG", "PG", "PF", "PG", "SG", "PF", "C", "PG", "SG", "SG", "PG", "SG", "SG", "SF", "PG", "PF", "SG", "C", "SG", "PF", "PG", "SF", "PF", "C"],
    "SCHOOL": ["San Francisco", "Duke", "null", "Marquette", "Kansas", "Spain", "Sweden", "UC Santa Barbara", "Washington State", "Marquette", "UCLA", "Colorado", "Serbia", "Arizona", "Houston", "Minnesota", "Kentucky", "North Carolina", "UConn", "Akron", "France", "Boston College", "UConn", "Gonzaga", "USC", "Kansas", "Cameroon", "Germany"]
}

# Create DataFrame
df = pd.DataFrame(data)



# Display the DataFrame
print(df)


# Save to CSV
file_path = 'C:/Users/Chuks/Documents/nba draft/2024_NBA_Draft_results2.csv'
df.to_csv(file_path, index=False)

file_path
