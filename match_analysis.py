import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_match_data(df):
    # Create features for head-to-head analysis
    match_features = [
        'xg', 'xga', 'npxg', 'npxga', 'deep', 'deep_allowed',
        'ppda_coef', 'oppda_coef', 'scored', 'missed'
    ]
    
    # Create form-based features
    df['recent_scoring_form'] = df.groupby('team')['scored'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['recent_defense_form'] = df.groupby('team')['missed'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['xg_form'] = df.groupby('team')['xg'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    return df

def predict_match_outcome(team1_data, team2_data, model):
    # Prepare match features for home team only to match training data
    match_features = team1_data[['xg_form', 'recent_scoring_form', 'recent_defense_form', 'ppda_coef']].values
    
    # Make prediction
    prediction = model.predict_proba([match_features])[0]
    
    return {
        'home_win': prediction[2],
        'draw': prediction[1],
        'away_win': prediction[0]
    }

def create_head_to_head_stats(df, team1, team2):
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Get all matches for both teams
    team1_matches = df_copy[df_copy['team'] == team1]
    team2_matches = df_copy[df_copy['team'] == team2]
    
    # Calculate head-to-head statistics
    stats = {
        'total_matches': len(team1_matches) + len(team2_matches),
        'goals_scored': {
            team1: team1_matches['scored'].sum(),
            team2: team2_matches['scored'].sum()
        },
        'average_xg': {
            team1: team1_matches['xg'].mean(),
            team2: team2_matches['xg'].mean()
        },
        'wins': {
            team1: len(team1_matches[team1_matches['scored'] > team1_matches['missed']]),
            team2: len(team2_matches[team2_matches['scored'] > team2_matches['missed']])
        },
        'draws': len(team1_matches[team1_matches['scored'] == team1_matches['missed']]) +
                len(team2_matches[team2_matches['scored'] == team2_matches['missed']])
    }
    
    return stats

def plot_head_to_head_comparison(stats, team1, team2):
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot wins comparison
    wins_data = [stats['wins'][team1], stats['draws'], stats['wins'][team2]]
    labels = [team1, 'Draws', team2]
    ax1.pie(wins_data, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Head-to-Head Results Distribution')
    
    # Plot goals and xG comparison
    metrics = ['Goals Scored', 'Average xG']
    team1_data = [stats['goals_scored'][team1], stats['average_xg'][team1]]
    team2_data = [stats['goals_scored'][team2], stats['average_xg'][team2]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, team1_data, width, label=team1)
    ax2.bar(x + width/2, team2_data, width, label=team2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_title('Performance Comparison')
    
    plt.tight_layout()
    return fig