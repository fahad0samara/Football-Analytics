import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from football_analysis import load_data, create_advanced_features, preprocess_data, train_model

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title
st.title('Performance Analysis')

# Load data
@st.cache_data
def load_processed_data():
    df = load_data()
    df = create_advanced_features(df)
    return df

df = load_processed_data()

# Team selection
selected_team = st.selectbox('Select Team for Analysis', options=sorted(df['team'].unique()))

# Recent Form Analysis
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Recent Form Analysis</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #424242'>Recent Form</h3>
    </div>
    """, unsafe_allow_html=True)
    recent_form = df[df['team'] == selected_team].tail(5)[['matches', 'pts']]
    st.line_chart(recent_form.set_index('matches'), use_container_width=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #424242'>Performance Factors</h3>
    </div>
    """, unsafe_allow_html=True)
    if st.button('Analyze Performance Factors'):
        with st.spinner('Analyzing performance factors...'):
            X_train, X_test, y_train, y_test, features = preprocess_data(df)
            model = train_model(X_train, y_train)
            
            feature_names = ['Expected Goals (xG)', 'Expected Goals Against (xGA)', 
                            'PPDA Coefficient', 'Opposition PPDA', 
                            'Deep Completions', 'Deep Completions Allowed',
                            'xG Difference', 'xGA Difference',
                            'Non-Penalty xG', 'Non-Penalty xGA']
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=range(len(importances)), y=importances[indices], palette='Blues_r')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Key Performance Indicators')
            plt.tight_layout()
            st.pyplot(fig)

# Team Comparison Section
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Team Comparison</h2>
</div>
""", unsafe_allow_html=True)

compare_team = st.selectbox('Select Team to Compare', 
                          options=[t for t in sorted(df['team'].unique()) if t != selected_team],
                          key='compare_team')

# Create comparison metrics
team1_data = df[df['team'] == selected_team].iloc[-1]
team2_data = df[df['team'] == compare_team].iloc[-1]

comparison_cols = ['pts', 'scored', 'missed', 'xg', 'xga', 'ppda_coef']
comparison_df = pd.DataFrame({
    selected_team: [team1_data[col] for col in comparison_cols],
    compare_team: [team2_data[col] for col in comparison_cols]
}, index=['Points', 'Goals Scored', 'Goals Conceded', 'Expected Goals', 'Expected Goals Against', 'PPDA'])

st.bar_chart(comparison_df)

# Advanced Metrics Analysis
st.markdown("""
<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Advanced Metrics Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Create advanced metrics visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot xG vs Actual Goals
ax1.scatter(team1_data['xg'], team1_data['scored'], label=selected_team)
ax1.scatter(team2_data['xg'], team2_data['scored'], label=compare_team)
ax1.plot([0, max(team1_data['xg'], team2_data['xg'])], 
         [0, max(team1_data['xg'], team2_data['xg'])], 
         'k--', alpha=0.5)
ax1.set_xlabel('Expected Goals (xG)')
ax1.set_ylabel('Actual Goals')
ax1.set_title('Expected vs Actual Goals')
ax1.legend()

# Plot Defensive Efficiency
ax2.scatter(team1_data['xga'], team1_data['missed'], label=selected_team)
ax2.scatter(team2_data['xga'], team2_data['missed'], label=compare_team)
ax2.plot([0, max(team1_data['xga'], team2_data['xga'])], 
         [0, max(team1_data['xga'], team2_data['xga'])], 
         'k--', alpha=0.5)
ax2.set_xlabel('Expected Goals Against (xGA)')
ax2.set_ylabel('Goals Conceded')
ax2.set_title('Expected vs Actual Goals Conceded')
ax2.legend()

plt.tight_layout()
st.pyplot(fig)