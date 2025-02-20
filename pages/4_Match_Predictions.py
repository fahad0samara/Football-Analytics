import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from football_analysis import load_data, create_advanced_features
from match_analysis import prepare_match_data, predict_match_outcome, create_head_to_head_stats, plot_head_to_head_comparison

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title
st.title('Match Predictions & Analysis')

# Load and process data
@st.cache_data
def load_match_data():
    df = load_data()
    df = create_advanced_features(df)
    df = prepare_match_data(df)
    return df

df = load_match_data()

# Match Prediction Section
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Head-to-Head Prediction</h2>
    <p style='color: #757575; margin-top: 1rem'>Select two teams to analyze their potential match outcome based on historical data and current form.</p>
</div>
""", unsafe_allow_html=True)

# Team Selection
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox('Select Home Team', options=sorted(df['team'].unique()), key='home_team')

with col2:
    away_team = st.selectbox('Select Away Team', 
                            options=[t for t in sorted(df['team'].unique()) if t != home_team],
                            key='away_team')

# Get team data
home_team_data = df[df['team'] == home_team].iloc[-1]
away_team_data = df[df['team'] == away_team].iloc[-1]

# Historical Head-to-Head Analysis
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Historical Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Create and display head-to-head stats
h2h_stats = create_head_to_head_stats(df, home_team, away_team)

# Display head-to-head visualization
fig = plot_head_to_head_comparison(h2h_stats, home_team, away_team)
st.pyplot(fig)

# Form Comparison
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Recent Form Comparison</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Home Team Form
    home_recent = df[df['team'] == home_team].tail(5)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(len(home_recent)), home_recent['pts'], marker='o', label=home_team, color='#1e88e5')
    plt.title(f'{home_team} - Last 5 Matches')
    plt.xlabel('Matches')
    plt.ylabel('Points')
    plt.legend()
    st.pyplot(fig)

with col2:
    # Away Team Form
    away_recent = df[df['team'] == away_team].tail(5)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(len(away_recent)), away_recent['pts'], marker='o', label=away_team, color='#1976d2')
    plt.title(f'{away_team} - Last 5 Matches')
    plt.xlabel('Matches')
    plt.ylabel('Points')
    plt.legend()
    st.pyplot(fig)

# Key Performance Metrics Comparison
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Key Performance Metrics</h2>
</div>
""", unsafe_allow_html=True)

# Create radar chart for team comparison
metrics = ['Scoring', 'Defense', 'Possession', 'Form']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)

# Prepare data for radar chart
home_values = [
    home_team_data['recent_scoring_form'],
    1 - home_team_data['recent_defense_form'],
    home_team_data['ppda_coef'],
    home_team_data['recent_form']
]

away_values = [
    away_team_data['recent_scoring_form'],
    1 - away_team_data['recent_defense_form'],
    away_team_data['ppda_coef'],
    away_team_data['recent_form']
]

# Normalize values
max_values = np.maximum(home_values, away_values)
min_values = np.minimum(home_values, away_values)
home_values_norm = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    for val, min_val, max_val in zip(home_values, min_values, max_values)]
away_values_norm = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                    for val, min_val, max_val in zip(away_values, min_values, max_values)]

# Create radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, home_values_norm, 'o-', linewidth=2, label=home_team, color='#1e88e5')
ax.fill(angles, home_values_norm, alpha=0.25, color='#1e88e5')
ax.plot(angles, away_values_norm, 'o-', linewidth=2, label=away_team, color='#1976d2')
ax.fill(angles, away_values_norm, alpha=0.25, color='#1976d2')

# Set chart properties
ax.set_thetagrids(angles * 180/np.pi, metrics)
ax.set_title('Team Comparison Radar', pad=20)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

st.pyplot(fig)

# Match Prediction
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Match Prediction</h2>
</div>
""", unsafe_allow_html=True)

# Calculate prediction probabilities
prediction = predict_match_outcome(home_team_data, away_team_data, None)  # Model would be trained separately

# Display prediction results
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='prediction-card' style='text-align: center'>
        <h3 style='color: white; margin: 0'>{home_team} Win</h3>
        <div style='font-size: 2rem; margin: 1rem 0'>{prediction['home_win']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='prediction-card' style='text-align: center'>
        <h3 style='color: white; margin: 0'>Draw</h3>
        <div style='font-size: 2rem; margin: 1rem 0'>{prediction['draw']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='prediction-card' style='text-align: center'>
        <h3 style='color: white; margin: 0'>{away_team} Win</h3>
        <div style='font-size: 2rem; margin: 1rem 0'>{prediction['away_win']:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

# Prediction Factors
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h3 style='margin:0; color: #1e88e5'>Key Prediction Factors</h3>
    <p style='color: #757575; margin-top: 1rem'>The prediction model considers recent form, head-to-head history, and various performance metrics to generate match probabilities.</p>
</div>
""", unsafe_allow_html=True)