import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
def load_data():
    try:
        import os
        data_file = 'understat.com.csv'
        # Try different possible data file locations
        possible_paths = [
            data_file,  # Current directory
            os.path.join(os.path.dirname(__file__), data_file),  # Same directory as this script
            os.path.join(os.path.dirname(__file__), 'data', data_file),  # data subdirectory
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Convert column names to lowercase and remove any spaces
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                logging.info(f'Successfully loaded data from {file_path} with {len(df)} rows')
                return df
        
        raise FileNotFoundError(f'Could not find {data_file} in any of the expected locations: {possible_paths}')
    except Exception as e:
        logging.error(f'Error loading data: {str(e)}')
        raise

# Preprocess the data
def create_advanced_features(df):
    # Calculate rolling averages for key metrics
    df['xg_rolling_avg'] = df.groupby('team')['xg'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['xga_rolling_avg'] = df.groupby('team')['xga'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Calculate team form (based on recent points)
    df['recent_form'] = df.groupby('team')['pts'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    # Calculate efficiency metrics
    df['scoring_efficiency'] = df['scored'] / df['xg']
    df['defensive_efficiency'] = df['xga'] / df['missed']
    
    return df

def preprocess_data(df):
    # Select relevant features for analysis
    features = ['xg', 'xga', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed',
               'xg_diff', 'xga_diff', 'npxg', 'npxga', 'xg_rolling_avg',
               'xga_rolling_avg', 'recent_form', 'scoring_efficiency',
               'defensive_efficiency']  # Enhanced feature set
    target = 'pts'
    
    try:
        # Check for missing values and print summary
        missing_summary = df[features + [target]].isnull().sum()
        logging.info(f'Missing values summary:\n{missing_summary}')
        
        # Calculate percentage of missing values per row
        missing_percentage = df[features].isnull().sum(axis=1) / len(features) * 100
        
        # Remove rows with more than 50% missing values
        df = df[missing_percentage <= 50]
        
        # For remaining missing values, use median imputation
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        
        # Remove outliers using IQR method
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))]
        
        # Split features and target
        X = df[features]
        y = df[target]
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info(f'Data preprocessing completed successfully. Final dataset shape: {df.shape}')
        return X_train_scaled, X_test_scaled, y_train, y_test, features
    except Exception as e:
        logging.error(f'Error in data preprocessing: {str(e)}')
        raise

# Perform hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                             cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    logging.info(f'Best parameters found: {grid_search.best_params_}')
    return grid_search.best_estimator_

# Train the model
def train_model(X_train, y_train):
    try:
        # Create base models
        rf = tune_hyperparameters(X_train, y_train)
        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        
        # Create ensemble model
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb)
        ])
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='r2')
        logging.info(f'Cross-validation scores: {cv_scores}')
        logging.info(f'Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
        
        return ensemble
    except Exception as e:
        logging.error(f'Error in model training: {str(e)}')
        raise

# Analyze feature importance
def analyze_feature_importance(model, feature_names):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance for Team Performance')
        sns.barplot(x=range(len(importances)), y=importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Print feature importance ranking
        print('\nFeature Importance Ranking:')
        for i in indices:
            print(f'{feature_names[i]}: {importances[i]:.4f}')
    except Exception as e:
        logging.error(f'Error in feature importance analysis: {str(e)}')
        raise

def main():
    try:
        # Load and preprocess data
        df = load_data()
        
        # Create advanced features
        df = create_advanced_features(df)
        
        # Preprocess the enhanced dataset
        X_train, X_test, y_train, y_test, features = preprocess_data(df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Analyze feature importance
        feature_names = ['Expected Goals (xG)', 'Expected Goals Against (xGA)', 
                        'PPDA Coefficient', 'Opposition PPDA', 
                        'Deep Completions', 'Deep Completions Allowed',
                        'xG Difference', 'xGA Difference',
                        'Non-Penalty xG', 'Non-Penalty xGA']
        analyze_feature_importance(model, feature_names)
        
        # Make predictions and evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print('\nModel Performance:')
        print(f'Training R² Score: {train_score:.3f}')
        print(f'Testing R² Score: {test_score:.3f}')
        
        logging.info('Analysis completed successfully')
    except Exception as e:
        logging.error(f'Error in main execution: {str(e)}')
        raise

if __name__ == '__main__':
    main()