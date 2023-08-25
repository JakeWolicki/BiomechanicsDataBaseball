import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read data from the CSV file
data = pd.read_csv('Biomechanics Analysis - Metadata Raw (1).csv')

# Extract features and pitch speed from the data
features = data[['session_mass_kg', 'session_height_m', 'age_yrs', 'playing_level']]
pitch_speed_mph = data['pitch_speed_mph'].values

# Categorical column (playing_level) needs to be one-hot encoded
categorical_cols = ['playing_level']
numerical_cols = ['session_mass_kg', 'session_height_m', 'age_yrs']

# Create a preprocessor for numerical and categorical columns
numeric_transformer = Pipeline(steps=[('scaler', 'passthrough')])  # You can use StandardScaler or other scalers here
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Create a pipeline with the preprocessor and the linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(features, pitch_speed_mph)

# Create a grid of values for each feature
mass_values = np.linspace(min(features['session_mass_kg']), max(features['session_mass_kg']), num=10)
height_values = np.linspace(min(features['session_height_m']), max(features['session_height_m']), num=10)
age_values = np.linspace(min(features['age_yrs']), max(features['age_yrs']), num=10)
playing_levels = data['playing_level'].unique()

# Generate all possible combinations of feature values
combinations = product(mass_values, height_values, age_values, playing_levels)

# Find the combination that yields the highest predicted pitch speed
best_combination = None
best_pitch_speed = float('-inf')

for combination in combinations:
    combination_dict = {
        'session_mass_kg': combination[0],
        'session_height_m': combination[1],
        'age_yrs': combination[2],
        'playing_level': combination[3]
    }
    pitch_speed = model.predict(pd.DataFrame(combination_dict, index=[0]))
    if pitch_speed > best_pitch_speed:
        best_pitch_speed = pitch_speed
        best_combination = combination_dict

print("Best Combination:", best_combination)
print("Best Pitch Speed:", best_pitch_speed)
