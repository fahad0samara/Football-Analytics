import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from football_analysis import load_data, create_advanced_features

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title
st.title('Team Overview')

# Load data
@st.cache_data
def load_processed_data():
    df = load_data()
    df = create_advanced_features(df)
    return df

df = load_processed_data()

# Team selection
selected_team = st.selectbox('Select Team for Analysis', options=sorted(df['team'].unique()))

# Performance Metrics Section
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem'>
    <h2 style='margin:0; color: #1e88e5'>Team Performance Dashboard</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
team_data = df[df['team'] == selected_team].iloc[-1]

# Enhanced metric cards
with col1:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Points</h4>
    </div>
    """, unsafe_allow_html=True)
    st.metric('', f"{team_data['pts']:.0f}", f"{team_data['xpts_diff']:.1f}")
    
with col2:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Goals Scored</h4>
    </div>
    """, unsafe_allow_html=True)
    st.metric('', f"{team_data['scored']:.0f}", f"{team_data['xg_diff']:.1f}")
    
with col3:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Goals Conceded</h4>
    </div>
    """, unsafe_allow_html=True)
    st.metric('', f"{team_data['missed']:.0f}", f"{team_data['xga_diff']:.1f}")

# Season Progression
st.subheader('Season Progression')
season_data = df[df['team'] == selected_team]

# Create tabs for different metrics
metric_tabs = st.tabs(['Points', 'Goals', 'Expected Goals'])

with metric_tabs[0]:
    # Points progression
    fig_points = plt.figure(figsize=(10, 6))
    plt.plot(season_data['matches'], season_data['pts'], marker='o')
    plt.title('Points Accumulation Over Season')
    plt.xlabel('Matches')
    plt.ylabel('Points')
    st.pyplot(fig_points)

with metric_tabs[1]:
    # Goals progression
    fig_goals = plt.figure(figsize=(10, 6))
    plt.plot(season_data['matches'], season_data['scored'], label='Scored', marker='o')
    plt.plot(season_data['matches'], season_data['missed'], label='Conceded', marker='o')
    plt.title('Goals Scored vs Conceded Over Season')
    plt.xlabel('Matches')
    plt.ylabel('Goals')
    plt.legend()
    st.pyplot(fig_goals)

with metric_tabs[2]:
    # Expected goals progression
    fig_xg = plt.figure(figsize=(10, 6))
    plt.plot(season_data['matches'], season_data['xg'], label='xG', marker='o')
    plt.plot(season_data['matches'], season_data['xga'], label='xGA', marker='o')
    plt.title('Expected Goals (xG) vs Expected Goals Against (xGA)')
    plt.xlabel('Matches')
    plt.ylabel('Expected Goals')
    plt.legend()
    st.pyplot(fig_xg)