import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simple_pid import PID
import matplotlib.pyplot as plt

# --- Step 1: Load dataset ---
combined_df = pd.read_csv("synthetic_dataset.csv")
# combined_df = pd.read_csv("combined_dataset.csv")

# --- Step 2: Define features and target ---
# Only include pre-print settings and early sensor readings to avoid PID feedback leakage
feature_cols = ['layer_height', 'wall_thickness', 'infill_density', 'nozzle_temperature',
                'bed_temperature', 'print_speed', 'fan_speed', 'material', 'infill_pattern']

X = combined_df[feature_cols]
y = combined_df['tension_strength']  # Target

# --- Step 2b: Preprocessing ---
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Fill missing values
X = X.fillna(X.mean())

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train predictive model ---
baseline_model = RandomForestRegressor(n_estimators=200, random_state=42)  # Slightly more trees
baseline_model.fit(X_train, y_train)
y_pred_test = baseline_model.predict(X_test)

# --- Step 6: Evaluation metrics ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Baseline RMSE: {rmse:.3f}")
print(f"Baseline MAE: {mae:.3f}")
print(f"Baseline R²: {r2:.3f}")

tension_range = combined_df['tension_strength'].max() - combined_df['tension_strength'].min()
print("Range of tension_strength:", tension_range)

# --- Step 6b: Layer-wise % deviation ---
percent_deviation = 100 * (y_pred_test - y_test.values) / y_test.values
print(f"Mean % deviation: {percent_deviation.mean():.2f}%")

# --- Step 10: Feature Importance Visualization ---
importances = baseline_model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Tension Strength Prediction")
plt.gca().invert_yaxis()
plt.show()