# --- Step 1: Import all necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simple_pid import PID
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

print("Libraries loaded successfully.")

try:
    df = pd.read_csv("dataset.csv")
    print("Synthetic dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'synthetic_dataset.csv' not found. Please run the CTGAN generation script first.")
    exit()

target_cols = ['tension_strength', 'elongation']
feature_cols = [
    'layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
    'nozzle_temperature', 'bed_temperature', 'print_speed', 'material', 'fan_speed',
    'sensor1_raw_mean', 'sensor1_raw_max', 'sensor1_raw_min', 'sensor1_raw_std',
    'sensor2_raw_mean', 'sensor2_raw_max', 'sensor2_raw_min', 'sensor2_raw_std',
    'sensor3_raw_mean', 'sensor3_raw_max', 'sensor3_raw_min', 'sensor3_raw_std',
    'sensor4_raw_mean', 'sensor4_raw_max', 'sensor4_raw_min', 'sensor4_raw_std',
    'sensor5_raw_mean', 'sensor5_raw_max', 'sensor5_raw_min', 'sensor5_raw_std'
]

X = df[feature_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_cols = ['material', 'infill_pattern']
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

xgb_base = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist' 
)

multi_xgb = MultiOutputRegressor(xgb_base)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', multi_xgb)
])

print("\nTraining the XGBoost model...")
pipeline.fit(X_train, y_train)
print("XGBoost model training complete.")

y_pred = pipeline.predict(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

print("\n--- Model Performance Metrics (XGBoost) ---")
for col in target_cols:
    rmse = np.sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
    mae = mean_absolute_error(y_test[col], y_pred_df[col])
    r2 = r2_score(y_test[col], y_pred_df[col])
    
    percent_deviation = 100 * (y_pred_df[col] - y_test[col]) / y_test[col]
    mean_dev = percent_deviation.mean()
    max_dev = percent_deviation.max()
    min_dev = percent_deviation.min()

    print(f"\nTarget: {col}")
    print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    print(f"Mean % deviation: {mean_dev:.2f}%")
    print(f"Max % deviation: {max_dev:.2f}%")
    print(f"Min % deviation: {min_dev:.2f}%")


feature_names = numeric_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

importances = np.mean([
    est.feature_importances_ for est in pipeline.named_steps['regressor'].estimators_
], axis=0)

feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importances['feature'].head(15), feature_importances['importance'].head(15), color='skyblue')
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
plt.show()

target_tension = 22.0
num_steps = 50

Kp = 2.5
Ki = 0.5
Kd = 0.1

pid = PID(Kp, Ki, Kd, setpoint=target_tension) 
pid.output_limits = (180, 250)

current_nozzle_temp = 230
predicted_tension_history = [0] 
adjusted_nozzle_temp_history = [current_nozzle_temp]

sample_features = X_test.iloc[[0]].copy()
sample_features = pd.concat([sample_features] * num_steps, ignore_index=True)

print("\nSimulating PID feedback loop with XGBoost model...")
for i in range(num_steps):
    sample_features.loc[i, 'nozzle_temperature'] = current_nozzle_temp
    
    current_pred = pipeline.predict(sample_features.loc[[i]])[0][0]
    
    noisy_pred = current_pred + np.random.normal(0, 0.25)
    
    control_output = pid(noisy_pred)
    
    current_nozzle_temp = control_output
    
    predicted_tension_history.append(noisy_pred)
    adjusted_nozzle_temp_history.append(current_nozzle_temp)

print("Simulation complete.")

fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.plot([target_tension] * num_steps, '-', label='Target Tension Strength', color='blue')
ax1.plot(predicted_tension_history, 'x-', label='Predicted Tension Strength (with noise)', color='orange')
ax1.set_xlabel('Simulation Step')
ax1.set_ylabel('Tension Strength Value')
ax1.set_title('PID Controller in Action with XGBoost Model')
ax1.set_ylim(15, 35)
ax1.grid(True)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(adjusted_nozzle_temp_history, '--', label='PID-Adjusted Nozzle Temp', color='green')
ax2.set_ylabel('Nozzle Temperature (°C)')
ax2.legend(loc='upper right')

plt.show()