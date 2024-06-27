import json
import pandas as pd
import os

# Define the directory containing the JSON files
json_files_path = "data/journals/"

# Initialize an empty list to store data
data = []

# Load JSON data from each file and adjust the IDs
for i in range(1, 13):
    file_path = os.path.join(json_files_path, f"{i}.json")
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = json.load(file)
        for entry in file_data:
            base_id = entry['id'] + (i - 1) * 25
            # Process journals_before_study
            for journal in entry['journals_before_study']:
                data.append({
                    "user_id": base_id,
                    "journal": journal,
                    "before_after": 0
                })
            # Process journals_after_study
            for journal in entry['journals_after_study']:
                data.append({
                    "user_id": base_id,
                    "journal": journal,
                    "before_after": 1
                })

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)
df['id'] = df.index + 1

# Save the DataFrame to a CSV file
csv_file_path = os.path.join('data', 'csv', 'journal_data.csv')
df.to_csv(csv_file_path, index=False)

# Load the CSV into a DataFrame to verify
df_loaded = pd.read_csv(csv_file_path)
print(df_loaded.head())
