import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')

df = pd.read_csv("data/csv/predicted_sentiments.csv")
df = df.drop(columns=['user_id', 'before_after', 'id'])

negative_words = ["miss", "sad", "bad", "hate", "come", "realli", "make", "feel", "morn"]
positive_words = ["love", "thank", "think", "haha", "work", "one", "right", "today", "yes", "time", "awesome"]

positive_journals = df[df['positive_probability'] > 0.9]
negative_journals = df[df['positive_probability'] < 0.1]

stop_words = set(stopwords.words('english'))

# Add punctuation to stop words
stop_words.update(string.punctuation)

# Function to clean and filter words
def clean_and_filter_words(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return filtered_words


positive_journals['cleaned_words'] = positive_journals['journal'].apply(clean_and_filter_words)
all_words = positive_journals['cleaned_words'].explode().value_counts()

print()
print("MOST FREQUENT WORDS IN POSITIVE JOURNALS")
print(all_words.head(50))

print()
print("SELECTED WORDS IN POSITIVE JOURNALS")
for word in positive_words:
    count = 0
    for journal in positive_journals['journal']:
        count += journal.lower().split().count(word)
    print(word, count)


print()
print()
print()

negative_journals['cleaned_words'] = negative_journals['journal'].apply(clean_and_filter_words)
all_words = negative_journals['cleaned_words'].explode().value_counts()

print()
print("MOST FREQUENT WORDS IN NEGATIVE JOURNALS")
print(all_words.head(50))

print()
print("SELECTED WORDS IN NEGATIVE JOURNALS")
for word in negative_words:
    count = 0
    for journal in negative_journals['journal']:
        count += journal.lower().split().count(word)
    print(word, count)

