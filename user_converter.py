import json
import pandas as pd
import os

# Define the directory containing the JSON files
json_files_path = "data/users/"

# Initialize an empty list to store data
data = []

# Load JSON data from each file
for i in range(1, 13):
    file_path = os.path.join(json_files_path, f"{i}.json")
    with open(file_path, 'r') as file:
        file_data = json.load(file)
        for entry in file_data:
            entry['id'] = entry['id'] + (i - 1) * 25
        data.extend(file_data)

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# Convert nested dictionary in 'big_five_personality' column to separate columns
df['hobbies'] = df['hobbies'].apply(lambda x: ','.join(x).replace(' ', ''))

personality_df = df['big_five_personality'].apply(pd.Series)
df = pd.concat([df.drop(columns=['big_five_personality']), personality_df], axis=1)

# Save the DataFrame to a CSV file
csv_file_path = os.path.join('data', 'csv', 'users.csv')
df.to_csv(csv_file_path, index=False)

# Load the CSV into a DataFrame to verify
df_loaded = pd.read_csv(csv_file_path)
print(df_loaded.head())
