"""
Generates a synthetic dataset matching the Fitbit project schema.
Replace this with the real Fitbit_dataset (same column names) when available —
no other code needs to change.
"""
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
N = 3000

workout_types = ["Cardio", "Strength", "HIIT", "Yoga"]
experience_levels = ["Beginner", "Intermediate", "Advanced"]
genders = ["Male", "Female"]

# Base demographics
age = np.random.randint(18, 60, N)
gender = np.random.choice(genders, N, p=[0.52, 0.48])
height = np.round(np.random.normal(1.70, 0.09, N).clip(1.45, 2.05), 2)

# Weight correlated loosely with height + noise
base_weight = 22 * height**2  # BMI ~22 baseline
weight = np.round(base_weight + np.random.normal(0, 10, N), 1).clip(40, 150)
bmi = np.round(weight / (height**2), 1)

fat_pct = np.round(
    np.where(gender == "Male",
             np.random.normal(18, 6, N),
             np.random.normal(26, 6, N)).clip(5, 45), 1)

experience = np.random.choice(experience_levels, N, p=[0.4, 0.4, 0.2])
exp_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
exp_num = np.array([exp_map[e] for e in experience])

workout_type = np.random.choice(workout_types, N, p=[0.3, 0.3, 0.2, 0.2])
intensity_map = {"Cardio": 1.0, "HIIT": 1.3, "Strength": 0.9, "Yoga": 0.5}
intensity = np.array([intensity_map[w] for w in workout_type])

max_bpm = np.round(207 - 0.7 * age + np.random.normal(0, 5, N)).clip(140, 210)
resting_bpm = np.round(70 - 3 * exp_num + np.random.normal(0, 6, N)).clip(45, 90)
avg_bpm = np.round(
    resting_bpm + (max_bpm - resting_bpm) * (0.25 + 0.35 * intensity)
    + np.random.normal(0, 3, N)
).clip(resting_bpm + 5, max_bpm)

session_duration = np.round(
    (np.random.gamma(shape=4, scale=0.18, size=N) + 0.25 * intensity)
    * (1 + 0.1 * exp_num), 2
).clip(0.2, 2.5)

water_intake = np.round(
    1.5 + 0.5 * exp_num + np.random.normal(0, 0.6, N), 2
).clip(0.5, 5.0)

workout_freq = np.random.randint(1, 8, N)
workout_freq = np.clip(workout_freq + exp_num, 1, 7)

# Calories burned: physiologically-plausible formula + noise
calories = (
    (avg_bpm - resting_bpm) * session_duration * 9
    + weight * 0.15 * session_duration * 10
    + intensity * 40
    + exp_num * 15
    - fat_pct * 0.8
    + np.random.normal(0, 40, N)
)
calories = np.round(calories.clip(80, 1400), 1)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Weight (kg)": weight,
    "Height (m)": height,
    "BMI": bmi,
    "Fat_Percentage": fat_pct,
    "Max_BPM": max_bpm.astype(int),
    "Avg_BPM": avg_bpm.astype(int),
    "Resting_BPM": resting_bpm.astype(int),
    "Session_Duration (hours)": session_duration,
    "Workout_Type": workout_type,
    "Water_Intake (liters)": water_intake,
    "Workout_Frequency (days/week)": workout_freq,
    "Experience_Level": experience,
    "Calories_Burned": calories,
})

# Inject a few missing values / mild outliers, as real sensor data would have
missing_idx = np.random.choice(N, size=int(N * 0.02), replace=False)
df.loc[missing_idx, "Water_Intake (liters)"] = np.nan
outlier_idx = np.random.choice(N, size=15, replace=False)
df.loc[outlier_idx, "Calories_Burned"] *= 1.8

df.to_csv(os.path.join(DATA_DIR, "fitbit_dataset.csv"), index=False)
print(f"Saved {len(df)} rows to data/fitbit_dataset.csv")
print(df.head())
print(df.isna().sum())
