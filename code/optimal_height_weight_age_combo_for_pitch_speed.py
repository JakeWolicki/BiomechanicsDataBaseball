import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data from the CSV file
data = pd.read_csv('Biomechanics Analysis - Metadata Raw.csv')

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

# Predict pitch speed using the model
predicted_pitch_speeds = model.predict(features)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(pitch_speed_mph, predicted_pitch_speeds)
print("Mean Squared Error:", mse)

# Visualize results for session_mass_kg, session_height_m, age_yrs, playing_level
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Pitch Speed vs. Features')

axes[0, 0].scatter(features['session_mass_kg'], pitch_speed_mph, color='blue', label='Actual')
axes[0, 0].scatter(features['session_mass_kg'], predicted_pitch_speeds, color='red', label='Predicted')
axes[0, 0].set_xlabel('Session Mass (kg)')
axes[0, 0].set_ylabel('Pitch Speed (mph)')
axes[0, 0].set_title('Pitch Speed vs. Session Mass')
axes[0, 0].legend()

axes[0, 1].scatter(features['session_height_m'], pitch_speed_mph, color='blue', label='Actual')
axes[0, 1].scatter(features['session_height_m'], predicted_pitch_speeds, color='red', label='Predicted')
axes[0, 1].set_xlabel('Session Height (m)')
axes[0, 1].set_ylabel('Pitch Speed (mph)')
axes[0, 1].set_title('Pitch Speed vs. Session Height')
axes[0, 1].legend()

axes[1, 0].scatter(features['age_yrs'], pitch_speed_mph, color='blue', label='Actual')
axes[1, 0].scatter(features['age_yrs'], predicted_pitch_speeds, color='red', label='Predicted')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].set_ylabel('Pitch Speed (mph)')
axes[1, 0].set_title('Pitch Speed vs. Age')
axes[1, 0].legend()

playing_levels = data['playing_level'].unique()
playing_levels.sort()
axes[1, 1].set_xticks(range(len(playing_levels)))
axes[1, 1].set_xticklabels(playing_levels, rotation=45)
for level in playing_levels:
    axes[1, 1].scatter(features[features['playing_level'] == level]['playing_level'], pitch_speed_mph[features['playing_level'] == level],
                       color='blue', label=f'Actual - {level}')
    predicted_speeds = model.predict(features[features['playing_level'] == level])
    axes[1, 1].scatter(features[features['playing_level'] == level]['playing_level'], predicted_speeds,
                       color='red', label=f'Predicted - {level}')
axes[1, 1].set_xlabel('Playing Level')
axes[1, 1].set_ylabel('Pitch Speed (mph)')
axes[1, 1].set_title('Pitch Speed vs. Playing Level')
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
