import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from football_analysis import load_data, create_advanced_features

# Set page config and custom CSS
st.set_page_config(page_title='Football Analytics Hub', layout='wide')

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Dashboard header with modern design
st.title('⚽ Football Analytics Hub')
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem'>
    <h3 style='margin:0; color: #1e88e5'>Welcome to Football Analytics Hub</h3>
    <p style='color: #757575; margin-top: 1rem; font-size: 1.1rem'>Explore comprehensive team statistics, performance predictions, and in-depth analysis using advanced metrics and machine learning models.</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_processed_data():
    try:
        df = load_data()
        df = create_advanced_features(df)
        return df
    except FileNotFoundError:
        st.warning("⚠️ Using sample data for demonstration. Please add your own data file for full functionality.")
        # Load sample data
        sample_df = pd.read_csv('data/sample_understat.com.csv')
        sample_df = create_advanced_features(sample_df)
        return sample_df

try:
    df = load_processed_data()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Quick Overview Section with enhanced metrics
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>League Overview</h2>
</div>
""", unsafe_allow_html=True)

# Display enhanced statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='stat-value'>{}</div>
        <div class='stat-label'>Total Teams</div>
    </div>
    """.format(len(df['team'].unique())), unsafe_allow_html=True)

with col2:
    total_matches = len(df) // len(df['team'].unique())
    st.markdown("""
    <div class='metric-card'>
        <div class='stat-value'>{}</div>
        <div class='stat-label'>Total Matches</div>
    </div>
    """.format(total_matches), unsafe_allow_html=True)

with col3:
    total_goals = df['scored'].sum()
    avg_goals = total_goals / (total_matches * 2)
    st.markdown("""
    <div class='metric-card'>
        <div class='stat-value'>{:.2f}</div>
        <div class='stat-label'>Average Goals per Match</div>
    </div>
    """.format(avg_goals), unsafe_allow_html=True)

with col4:
    top_team = df.groupby('team')['scored'].sum().idxmax()
    top_goals = df.groupby('team')['scored'].sum().max()
    st.markdown("""
    <div class='metric-card'>
        <div class='stat-value'>{}</div>
        <div class='stat-label'>Top Scoring Team ({:.0f} goals)</div>
    </div>
    """.format(top_team, top_goals), unsafe_allow_html=True)

# League Performance Trends
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>League Performance Trends</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Goals Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='scored', bins=20, color='#1e88e5', alpha=0.7)
    plt.title('Goals Distribution Across All Matches')
    plt.xlabel('Goals Scored')
    plt.ylabel('Frequency')
    st.pyplot(fig)

with col2:
    # Team Performance Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='xg', y='scored', alpha=0.6, color='#1e88e5')
    plt.title('Expected vs Actual Goals')
    plt.xlabel('Expected Goals (xG)')
    plt.ylabel('Actual Goals')
    st.pyplot(fig)

# Team Form Analysis
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Team Form Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Add team selection for form analysis
selected_team = st.selectbox('Select Team', options=sorted(df['team'].unique()))
team_data = df[df['team'] == selected_team].tail(10)

col1, col2 = st.columns(2)

with col1:
    # Recent Form Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(len(team_data)), team_data['pts'], marker='o', color='#1e88e5')
    plt.title(f'{selected_team} - Recent Form')
    plt.xlabel('Last 10 Matches')
    plt.ylabel('Points')
    st.pyplot(fig)

with col2:
    # Performance Metrics
    recent_stats = {
        'Average Points': team_data['pts'].mean(),
        'Goals Scored': team_data['scored'].sum(),
        'Goals Conceded': team_data['missed'].sum(),
        'xG Performance': team_data['xg'].mean()
    }
    
    for metric, value in recent_stats.items():
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom: 1rem'>
            <div class='stat-value'>{value:.2f}</div>
            <div class='stat-label'>{metric}</div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced League Overview Section
st.markdown("""
<div style='background-color: #2d2d2d; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); margin: 2rem 0'>
    <h2 style='margin:0; color: #2196f3'>League Overview</h2>
</div>
""", unsafe_allow_html=True)

# Add Advanced Team Statistics
st.markdown("""
<div style='background-color: #2d2d2d; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); margin: 2rem 0'>
    <h2 style='margin:0; color: #2196f3'>Advanced Team Statistics</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Team Performance Radar Chart
    selected_team_stats = df[df['team'] == selected_team].iloc[-1]
    stats_labels = ['Scoring', 'Defense', 'Possession', 'Expected Goals']
    stats_values = [
        selected_team_stats['scored'] / df['scored'].max(),
        1 - (selected_team_stats['missed'] / df['missed'].max()),
        selected_team_stats['ppda_coef'] / df['ppda_coef'].max(),
        selected_team_stats['xg'] / df['xg'].max()
    ]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2*np.pi, len(stats_labels), endpoint=False)
    stats_values = np.concatenate((stats_values, [stats_values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, stats_values, 'o-', linewidth=2, color='#2196f3')
    ax.fill(angles, stats_values, alpha=0.25, color='#2196f3')
    ax.set_thetagrids(angles[:-1] * 180/np.pi, stats_labels)
    ax.set_title('Team Performance Radar', color='white', pad=20)
    ax.grid(True, color='gray', alpha=0.5)
    ax.spines['polar'].set_color('gray')
    ax.set_facecolor('#2d2d2d')
    fig.patch.set_facecolor('#2d2d2d')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Detailed Statistics Table
    detailed_stats = pd.DataFrame({
        'Metric': ['Goals per Game', 'Clean Sheets', 'Goal Difference', 'xG Ratio', 'Possession Score'],
        'Value': [
            selected_team_stats['scored'] / len(df[df['team'] == selected_team]),
            len(df[(df['team'] == selected_team) & (df['missed'] == 0)]),
            selected_team_stats['scored'] - selected_team_stats['missed'],
            selected_team_stats['xg'] / selected_team_stats['xga'],
            selected_team_stats['ppda_coef']
        ]
    })
    
    for _, row in detailed_stats.iterrows():
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom: 1rem'>
            <div class='stat-value'>{row['Value']:.2f}</div>
            <div class='stat-label'>{row['Metric']}</div>
        </div>
        """, unsafe_allow_html=True)

# Navigation Guide with enhanced styling
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Navigation Guide</h2>
    <p style='color: #757575; margin-top: 1rem; font-size: 1.1rem'>Use the sidebar to navigate through different sections:</p>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem'>
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px'>
            <h4 style='color: #1e88e5; margin: 0'>Team Overview</h4>
            <p style='color: #757575; margin-top: 0.5rem'>View detailed performance metrics for each team</p>
        </div>
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px'>
            <h4 style='color: #1e88e5; margin: 0'>Performance Analysis</h4>
            <p style='color: #757575; margin-top: 0.5rem'>Analyze team performance factors and comparisons</p>
        </div>
        <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px'>
            <h4 style='color: #1e88e5; margin: 0'>League Standings</h4>
            <p style='color: #757575; margin-top: 0.5rem'>Check current league table and team statistics</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)