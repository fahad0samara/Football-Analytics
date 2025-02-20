import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from football_analysis import load_data, create_advanced_features

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title and description
st.title('Team Benchmarking & Analysis')
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Advanced Team Benchmarking</h2>
    <p style='color: #757575; margin-top: 1rem'>Compare team performance against league averages and analyze historical trends using advanced metrics.</p>
</div>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_processed_data():
    df = load_data()
    df = create_advanced_features(df)
    return df

df = load_processed_data()

# Team selection
selected_team = st.selectbox('Select Team for Benchmarking', options=sorted(df['team'].unique()))

# Calculate league averages and team metrics
league_avg = df.groupby('matches').agg({
    'scored': 'mean',
    'missed': 'mean',
    'xg': 'mean',
    'xga': 'mean',
    'ppda_coef': 'mean',
    'pts': 'mean'
}).mean()

team_stats = df[df['team'] == selected_team].iloc[-1]

# Performance Benchmarking Section
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Performance Benchmarking</h2>
</div>
""", unsafe_allow_html=True)

# Create benchmark comparison
col1, col2 = st.columns(2)

with col1:
    # Radar Chart for Team vs League Average
    metrics = ['Scoring', 'Defense', 'Expected Goals', 'Expected Goals Against', 'Pressing Efficiency']
    team_values = [
        team_stats['scored'],
        -team_stats['missed'],
        team_stats['xg'],
        -team_stats['xga'],
        team_stats['ppda_coef']
    ]
    league_values = [
        league_avg['scored'],
        -league_avg['missed'],
        league_avg['xg'],
        -league_avg['xga'],
        league_avg['ppda_coef']
    ]
    
    # Normalize values
    team_normalized = (team_values - np.min(team_values)) / (np.max(team_values) - np.min(team_values))
    league_normalized = (league_values - np.min(league_values)) / (np.max(league_values) - np.min(league_values))
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, team_normalized, 'o-', linewidth=2, label=selected_team, color='#1e88e5')
    ax.fill(angles, team_normalized, alpha=0.25, color='#1e88e5')
    ax.plot(angles, league_normalized, 'o-', linewidth=2, label='League Average', color='#ff9800')
    ax.fill(angles, league_normalized, alpha=0.25, color='#ff9800')
    
    ax.set_thetagrids(angles * 180/np.pi, metrics)
    ax.set_title('Team vs League Average Performance', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    st.pyplot(fig)

with col2:
    # Performance Metrics Table
    metrics_comparison = pd.DataFrame({
        'Metric': ['Goals Scored', 'Goals Conceded', 'Expected Goals', 'Expected Goals Against', 'PPDA'],
        'Team': [team_stats['scored'], team_stats['missed'], team_stats['xg'], team_stats['xga'], team_stats['ppda_coef']],
        'League Avg': [league_avg['scored'], league_avg['missed'], league_avg['xg'], league_avg['xga'], league_avg['ppda_coef']],
        'Difference': [
            team_stats['scored'] - league_avg['scored'],
            team_stats['missed'] - league_avg['missed'],
            team_stats['xg'] - league_avg['xg'],
            team_stats['xga'] - league_avg['xga'],
            team_stats['ppda_coef'] - league_avg['ppda_coef']
        ]
    })
    
    for idx, row in metrics_comparison.iterrows():
        st.markdown(f"""
        <div class='benchmark-card'>
            <h3 style='margin:0'>{row['Metric']}</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem'>
                <div>
                    <div style='font-size: 1.5rem; font-weight: 700'>{row['Team']:.2f}</div>
                    <div style='font-size: 0.9rem; opacity: 0.8'>Team</div>
                </div>
                <div>
                    <div style='font-size: 1.5rem; font-weight: 700'>{row['League Avg']:.2f}</div>
                    <div style='font-size: 0.9rem; opacity: 0.8'>League Avg</div>
                </div>
                <div>
                    <div style='font-size: 1.5rem; font-weight: 700'>{row['Difference']:.2f}</div>
                    <div style='font-size: 0.9rem; opacity: 0.8'>Difference</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Historical Trends Section
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Historical Performance Trends</h2>
</div>
""", unsafe_allow_html=True)

# Get historical data for the selected team
team_history = df[df['team'] == selected_team].copy()
team_history['rolling_pts'] = team_history['pts'].rolling(window=5, min_periods=1).mean()
team_history['rolling_xg'] = team_history['xg'].rolling(window=5, min_periods=1).mean()

# Create trend visualizations
col1, col2 = st.columns(2)

with col1:
    # Points Trend
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(team_history['matches'], team_history['rolling_pts'], color='#1e88e5', label='Rolling Average')
    plt.scatter(team_history['matches'], team_history['pts'], color='#1e88e5', alpha=0.5, label='Actual Points')
    plt.title('Points Trend Over Season')
    plt.xlabel('Matches')
    plt.ylabel('Points')
    plt.legend()
    st.pyplot(fig)

with col2:
    # Expected Goals Trend
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(team_history['matches'], team_history['rolling_xg'], color='#1e88e5', label='Rolling xG')
    plt.scatter(team_history['matches'], team_history['xg'], color='#1e88e5', alpha=0.5, label='Actual xG')
    plt.title('Expected Goals (xG) Trend')
    plt.xlabel('Matches')
    plt.ylabel('Expected Goals')
    plt.legend()
    st.pyplot(fig)

# Performance Insights
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Performance Insights</h2>
</div>
""", unsafe_allow_html=True)

# Calculate performance insights
performance_vs_xg = team_stats['scored'] / team_stats['xg']
defensive_efficiency = team_stats['missed'] / team_stats['xga']
points_per_xg = team_stats['pts'] / team_stats['xg']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #1e88e5'>Scoring Efficiency</h3>
        <div style='font-size: 2rem; font-weight: 700; color: #1e88e5'>{performance_vs_xg:.2f}</div>
        <div style='color: #757575'>Goals per Expected Goal</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #1e88e5'>Defensive Efficiency</h3>
        <div style='font-size: 2rem; font-weight: 700; color: #1e88e5'>{defensive_efficiency:.2f}</div>
        <div style='color: #757575'>Goals Conceded per xGA</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #1e88e5'>Points Efficiency</h3>
        <div style='font-size: 2rem; font-weight: 700; color: #1e88e5'>{points_per_xg:.2f}</div>
        <div style='color: #757575'>Points per Expected Goal</div>
    </div>
    """, unsafe_allow_html=True)