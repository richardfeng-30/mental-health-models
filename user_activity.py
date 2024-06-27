import random
import numpy as np
import pandas as pd


df = pd.read_csv("data/csv/user_scores_lstm.csv")

print(df.columns.tolist()) 


df['mental_health_score_before'] = df['average_positive_probability_before'] * 100
df['mental_health_score_after'] = df['average_positive_probability_after'] * 100

df['score_change'] = df['mental_health_score_after'] - df['mental_health_score_before']



def positive1(score_change):
    if score_change > 20:
        return random.randint(60, 150)
    elif score_change > 0:
        return random.randint(30, 120)
    else:
        return random.randint(30, 120)
    
def positive2(score_change):
    if score_change > 20:
        return random.randint(90, 150)
    elif score_change > 0:
        return random.randint(60, 90)
    else:
        return random.randint(0, 60)

def positive3(score_change):
    if score_change > 20:
        return random.randint(120, 150)
    elif score_change > 0:
        return random.randint(60, 120)
    else:
        return random.randint(0, 60)

def neutralLow(score_change):
    return random.randint(0, 30)

def neutralHigh(score_change):
    return random.randint(30, 150)

def negative1(score_change):
    if score_change < -20:
        return random.randint(90, 150)
    elif score_change < 0:
        return random.randint(60, 90)
    else:
        return random.randint(30, 60)
    
def calculate_journals_written(score_change):
    if score_change > 20:
        return random.randint(120, 360)
    elif score_change > 0:
        return random.randint(90, 240)
    else:
        return random.randint(0, 90)
    
def calculate_avg_sleep(score_change):
    if score_change > 20:
        return round(random.uniform(9, 11), 1)
    elif score_change > 0:
        return round(random.uniform(8, 10), 1)
    else:
        return round(random.uniform(6, 9), 1)


df['gratitude'] = df['score_change'].apply(positive3)
df['journals'] = df['score_change'].apply(calculate_journals_written)

df['sunny'] = df['score_change'].apply(positive1)
df['cloudy'] = df['score_change'].apply(neutralLow)
df['rainy'] = df['score_change'].apply(negative1)
df['snowy'] = df['score_change'].apply(neutralLow)
df['windy'] = df['score_change'].apply(neutralLow)

df['exercise'] = df['score_change'].apply(positive3)
df['movie_tv'] = df['score_change'].apply(negative1)
df['gaming'] = df['score_change'].apply(negative1)
df['reading'] = df['score_change'].apply(positive1)
df['instrument'] = df['score_change'].apply(positive2)
df['walk'] = df['score_change'].apply(positive1)
df['music'] = df['score_change'].apply(positive3)
df['drawing'] = df['score_change'].apply(positive1)

df['class'] = df['score_change'].apply(neutralHigh)
df['study'] = df['score_change'].apply(neutralHigh)
df['homework'] = df['score_change'].apply(negative1)
df['exam'] = df['score_change'].apply(negative1)

df['sleep'] = df['score_change'].apply(calculate_avg_sleep)

df.to_csv('data/csv/user_data.csv', index=False)

print(df.head)
