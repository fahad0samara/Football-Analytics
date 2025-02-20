import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from football_analysis import load_data, create_advanced_features, preprocess_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import centralized CSS
with open('static/styles.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page title and description
st.title('Advanced Team Analytics')
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Performance Analytics Dashboard</h2>
    <p style='color: #757575; margin-top: 1rem'>Advanced statistical analysis and machine learning insights for team performance evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_processed_data():
    df = load_data()
    df = create_advanced_features(df)
    return df

df = load_processed_data()

# Team Selection
selected_team = st.selectbox('Select Team for Analysis', options=sorted(df['team'].unique()))

# Performance Metrics Section
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Advanced Performance Metrics</h2>
</div>
""", unsafe_allow_html=True)

# Calculate advanced metrics
team_data = df[df['team'] == selected_team]
recent_data = team_data.tail(5)

# Performance Trends
col1, col2 = st.columns(2)

with col1:
    # Expected Goals Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(team_data['xg'], team_data['scored'], alpha=0.6)
    plt.plot([0, max(team_data['xg'])], [0, max(team_data['xg'])], '--', color='gray')
    plt.title('Expected vs Actual Goals')
    plt.xlabel('Expected Goals (xG)')
    plt.ylabel('Actual Goals')
    st.pyplot(fig)

with col2:
    # Form Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(recent_data['matches'], recent_data['recent_form'], marker='o')
    plt.title('Recent Form Trend')
    plt.xlabel('Matches')
    plt.ylabel('Form Rating')
    st.pyplot(fig)

# Advanced Analytics Section
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Machine Learning Insights</h2>
</div>
""", unsafe_allow_html=True)

# Prepare data for ML analysis
X_train, X_test, y_train, y_test, features = preprocess_data(df)
model = train_model(X_train, y_train)

# Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Display feature importance
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Performance Prediction')
plt.xlabel('Importance Score')
st.pyplot(fig)

# Performance Prediction
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Performance Prediction</h2>
</div>
""", unsafe_allow_html=True)

# Get latest team stats for prediction
latest_stats = team_data.iloc[-1][features]
scaler = StandardScaler()
scaled_stats = scaler.fit_transform(latest_stats.values.reshape(1, -1))
predicted_points = model.predict(scaled_stats)[0]

# Display prediction
st.markdown(f"""
<div class='analysis-card' style='text-align: center'>
    <h3 style='color: white; margin: 0'>Predicted Points Next Match</h3>
    <div style='font-size: 2rem; margin: 1rem 0'>{predicted_points:.2f}</div>
</div>
""", unsafe_allow_html=True)

# Model Performance Metrics
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Model Performance</h2>
</div>
""", unsafe_allow_html=True)

# Calculate model metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #1e88e5'>R² Score</h3>
        <div style='font-size: 2rem; font-weight: 700; color: #1e88e5'>{r2:.3f}</div>
        <div style='color: #757575'>Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #1e88e5'>Mean Squared Error</h3>
        <div style='font-size: 2rem; font-weight: 700; color: #1e88e5'>{mse:.3f}</div>
        <div style='color: #757575'>Prediction Error</div>
    </div>
    """, unsafe_allow_html=True)

# Historical Performance Analysis
st.markdown("""
<div style='background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0'>
    <h2 style='margin:0; color: #1e88e5'>Historical Performance Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Calculate historical trends
team_data['rolling_xg'] = team_data['xg'].rolling(window=5).mean()
team_data['rolling_pts'] = team_data['pts'].rolling(window=5).mean()

# Plot historical performance
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(team_data['matches'], team_data['rolling_xg'], label='Rolling xG')
plt.plot(team_data['matches'], team_data['rolling_pts'], label='Rolling Points')
plt.title('Historical Performance Trends')
plt.xlabel('Matches')
plt.ylabel('Performance Metrics')
plt.legend()
st.pyplot(fig)