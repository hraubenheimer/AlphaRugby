import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import sys
import test
import warnings
warnings.filterwarnings("ignore")

start_date = sys.argv[1]
if len(sys.argv) > 2:
    end_date = sys.argv[2]

df = pd.read_csv('Data/data.csv')
pd.set_option('display.max_rows', None)

df = df[df['date'] > '2003-10-06']

def drawToLoss(row):
    if row['result'] == 0:
        row['result'] = 1
    return row
df = df.apply(drawToLoss, axis=1)

def find_unex(row):
    return row['result']*abs(row['rank_diff']) if row['rank_diff']*row['result'] <= 0 else 0
df['unexpectedness'] = df.apply(find_unex, axis=1)

def find_h2h_unex(row):
    back = 5
    new_df = df[(df.index < row.name) & (((df['team1'] == row['team1']) & (df['team2'] == row['team2'])) | ((df['team2'] == row['team1']) & (df['team1'] == row['team2'])))]
    new_df.loc[new_df['team1'] == row['team2'], 'unexpectedness'] *= -1
    results = list(new_df['unexpectedness'][-back:])
    weights = [1]*back
    if len(results):
        largest_weights_desc = sorted(weights, reverse=True)[:len(results)]
        largest_weights_asc = sorted(largest_weights_desc)
        weights = largest_weights_asc
    for i in range(len(results)):
        results[i] = results[i]*weights[i]
    return sum(results)

def find_h2h_score(row):
    back = 5
    new_df = df[(df.index < row.name) & (((df['team1'] == row['team1']) & (df['team2'] == row['team2'])) | ((df['team2'] == row['team1']) & (df['team1'] == row['team2'])))]
    new_df.loc[new_df['team1'] == row['team2'], 'score_diff'] *= -1
    results = list(new_df['score_diff'][-back:])
    weights = np.linspace(0, 1, back)
    if len(results):
        largest_weights_desc = sorted(weights, reverse=True)[:len(results)]
        largest_weights_asc = sorted(largest_weights_desc)
        weights = largest_weights_asc
    for i in range(len(results)):
        results[i] = results[i]*weights[i]
    return sum(results)

def find_h2h_score2(row):
    back = 15
    new_df = df[(df.index < row.name) & (((df['team1'] == row['team1']) & (df['team2'] == row['team2'])) | ((df['team2'] == row['team1']) & (df['team1'] == row['team2'])))]
    new_df.loc[new_df['team1'] == row['team2'], 'score_diff'] *= -1
    results = list(new_df['score_diff'][-back:])
    weights = np.linspace(0, 1, back)
    if len(results):
        largest_weights_desc = sorted(weights, reverse=True)[:len(results)]
        largest_weights_asc = sorted(largest_weights_desc)
        weights = largest_weights_asc
    for i in range(len(results)):
        results[i] = results[i]*weights[i]
    return sum(results)

df['h2h'] = df.apply(find_h2h_unex, axis=1)
df['h2h_score'] = df.apply(find_h2h_score, axis=1)
df['h2h_score_2'] = df.apply(find_h2h_score2, axis=1)

df1 = df[df['neutral']==0]
move_rd = (df1[df1['result'] == 1]['rank_diff'].mean()+df1[df1['result'] == -1]['rank_diff'].mean())
move_sd = (df1[df1['result'] == 1]['score_diff'].mean()+df1[df1['result'] == -1]['score_diff'].mean())
def makeNeutralRD(row):
    if row['neutral'] == 0:
        row['rank_diff'] = row['rank_diff'] + move_rd
    return row
def makeHomeRD(row):
    if row['neutral'] == 0:
        row['rank_diff'] = row['rank_diff'] - move_rd
    return row
def makeNeutralSD(row):
    if row['neutral'] == 0:
        row['score_diff'] = row['score_diff'] + move_sd
    return row
def makeHomeSD(row):
    if row['neutral'] == 0:
        row['score_diff'] = row['score_diff'] - move_sd
    return row

def processScores(row):
    if row['result'] == -1:
        row['rank_diff'] = -row['rank_diff']
        row['score_diff'] = -row['score_diff']
    return row
df1 = df.apply(processScores, axis=1)

def scaleScores(row):
    if abs(row['rank_diff']) >= 15:
        row['predicted_score'] *= 1.5
    return row

probs = []
def winner(row):
    if row['result'] == 1:
        row['Winner'] = row['team1']
        row['Probability of Winning'] = str(round(probs[row.name][1]*100, 2))+'%'
    else:
        row['Winner'] = row['team2']
        row['Probability of Winning'] = str(round(probs[row.name][0]*100, 2))+'%'
    return row

df.dropna(inplace=True)
matches = df[df['date'] >= start_date].copy()
if len(sys.argv) > 2:
    matches = matches[matches['date'] <= end_date].copy()
df = df[df['date'] < start_date].copy()

result_features = ['rank_diff', 'h2h_score', 'h2h']
score_features = ['rank_diff', 'h2h_score_2']

logreg = LogisticRegression(fit_intercept=False)
train = df.apply(makeNeutralRD, axis=1)
tour = matches.apply(makeHomeRD, axis=1)
logreg.fit(train[result_features], train['result'])
probs = logreg.predict_proba(tour[result_features])
matches.reset_index(inplace=True)
matches['result'] = logreg.predict(tour[result_features])

linreg = LinearRegression()
train = df.apply(makeNeutralSD, axis=1)
train = train.apply(processScores, axis=1) 
linreg.fit(train[score_features], train['score_diff'])
tour = matches.apply(processScores, axis=1)
matches['predicted_score'] = linreg.predict(tour[score_features])
matches = matches.apply(scaleScores, axis=1)
matches['Winning Margin'] = matches['predicted_score'].round().astype(int)

matches = matches.apply(winner, axis=1)
matches = matches.rename(columns={'date': 'Date', 'team1': 'Team 1', 'team2': 'Team 2'})
print(matches[['Date', 'Team 1', 'Team 2', 'Winner', 'Probability of Winning', 'Winning Margin']].to_string(index=False))