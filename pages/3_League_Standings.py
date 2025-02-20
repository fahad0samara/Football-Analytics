import streamlit as st
import pandas as pd
from football_analysis import load_data, create_advanced_features

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title
st.title('League Standings')

# Load data
@st.cache_data
def load_processed_data():
    df = load_data()
    df = create_advanced_features(df)
    return df

df = load_processed_data()

# League Table Section
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Current League Standings</h2>
</div>
""", unsafe_allow_html=True)

# Get the latest data for each team
latest_data = df.groupby('team').last().reset_index()
standings = latest_data[['team', 'pts', 'scored', 'missed', 'wins', 'draws', 'loses']]
standings = standings.sort_values('pts', ascending=False).reset_index(drop=True)
standings.index = standings.index + 1  # Start position from 1

# Calculate goal difference
standings['GD'] = standings['scored'] - standings['missed']

# Display standings with enhanced styling
st.dataframe(
    standings.rename(columns={
        'team': 'Team',
        'pts': 'Points',
        'scored': 'GF',
        'missed': 'GA',
        'wins': 'W',
        'draws': 'D',
        'loses': 'L'
    }),
    use_container_width=True
)

# Team Statistics Section
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Team Statistics</h2>
</div>
""", unsafe_allow_html=True)

# Add team selection for detailed stats
selected_team = st.selectbox('Select Team for Detailed Statistics', options=sorted(df['team'].unique()))

# Display team statistics
team_stats = latest_data[latest_data['team'] == selected_team].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>League Position</h4>
    </div>
    """, unsafe_allow_html=True)
    position = standings[standings['team'] == selected_team].index[0]
    st.metric('', f"{position}")

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Win Rate</h4>
    </div>
    """, unsafe_allow_html=True)
    win_rate = (team_stats['wins'] / (team_stats['wins'] + team_stats['draws'] + team_stats['loses'])) * 100
    st.metric('', f"{win_rate:.1f}%")

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Goals per Game</h4>
    </div>
    """, unsafe_allow_html=True)
    goals_per_game = team_stats['scored'] / (team_stats['wins'] + team_stats['draws'] + team_stats['loses'])
    st.metric('', f"{goals_per_game:.2f}")

with col4:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: #1e88e5; margin:0'>Clean Sheets</h4>
    </div>
    """, unsafe_allow_html=True)
    clean_sheets = len(df[(df['team'] == selected_team) & (df['missed'] == 0)])
    st.metric('', f"{clean_sheets}")

# Form Guide
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h3 style='margin:0; color: #1e88e5'>Recent Form</h3>
</div>
""", unsafe_allow_html=True)

# Get last 5 matches
recent_matches = df[df['team'] == selected_team].tail(5)
form_data = pd.DataFrame({
    'Match': range(1, 6),
    'Points': recent_matches['pts'],
    'Goals For': recent_matches['scored'],
    'Goals Against': recent_matches['missed']
})

st.dataframe(form_data, use_container_width=True)