import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'data/csv/predicted_sentiments_lstm.csv'
df = pd.read_csv(file_path)

grouped = df.groupby(['user_id', 'before_after'])
summary = grouped.agg(
    average_positive_probability=('positive_probability', 'mean'),
    count_id=('id', 'count')
).reset_index()

before = summary[summary['before_after'] == 0].drop(columns='before_after').rename(columns={
    'average_positive_probability': 'average_positive_probability_before',
    'count_id': 'count_id_before'
})
after = summary[summary['before_after'] == 1].drop(columns='before_after').rename(columns={
    'average_positive_probability': 'average_positive_probability_after',
    'count_id': 'count_id_after'
})

# Merge the before and after data on user_id
result = pd.merge(before, after, on='user_id', how='outer')

csv_file_path = "data/csv/agg_journals_lstm.csv"
result.to_csv(csv_file_path, index=False)


user_file = 'data/csv/users.csv'
user_df = pd.read_csv(user_file)

final_result = pd.merge(user_df, result, left_on='id', right_on='user_id', how='inner')
final_result = final_result.drop(columns=['user_id'])

print(final_result)

# Save the final result to a new CSV file if needed
final_result.to_csv('data/csv/user_scores_lstm.csv', index=False)
