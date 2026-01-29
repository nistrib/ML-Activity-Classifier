import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score)
import joblib

#region ---------------------------------CREATE HDF5 FILE-----------------------------------------------
# Create h5py file for organization - this allows structured storage of raw, preprocessed, and segmented data
with h5py.File("dataset.h5", "w") as f:
    # Create main data groups for different processing stages
    raw_data_group = f.create_group("Raw Data")
    preprocess_data_group = f.create_group("Pre-processed Data")
    segmented_data_group = f.create_group("Segmented Data")
    
    # Create subgroups for training and testing data
    segmented_train_data_group = segmented_data_group.create_group("Train")
    segmented_test_data_group = segmented_data_group.create_group("Test")
    
    # Create groups for each person's data
    for member in ["Lorenzo", "Kaykay", "Daniil"]:
        raw_data_group.create_group(member)
        preprocess_data_group.create_group(member)

# Create directories for different file types on disk
directories = ['raw', 'processed', 'segmented']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_missing_values(name, df):
    """Print missing values and dash-based missing markers in the dataframe."""
    print(f"{name} NaNs:\n{df.isna().sum()}")
    print(f"{name} dashes: {(df == '-').sum().sum()}\n")
#endregion

#region ---------------------------------LOAD RAW DATA AND STORE IN HDF5-----------------------------------------------
# Dictionary of raw data CSV files for each person and activity
raw_dfs = {
    "lorenzo_walking": pd.read_csv("raw/lorenzo_walking_raw.csv"),
    "lorenzo_jumping": pd.read_csv("raw/lorenzo_jumping_raw.csv"),
    "kaykay_walking": pd.read_csv("raw/kaykay_walking_raw.csv"),
    "kaykay_jumping": pd.read_csv("raw/kaykay_jumping_raw.csv"),
    "daniil_walking": pd.read_csv("raw/daniil_walking_raw.csv"),
    "daniil_jumping": pd.read_csv("raw/daniil_jumping_raw.csv")
}

# Store all raw data into the HDF5 file
with h5py.File("dataset.h5", "a") as f:
    for name, df in raw_dfs.items():
        # Extract person name and activity type from the dataset name
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        # Store raw data in HDF5
        f[f"Raw Data/{person}/{activity}"] = df.values
#endregion

#region ------------------------------PROCESS RAW DATA AND STORE IN HDF5-----------------------------------------------  
def preprocess_dataframe(df, window_size=51):
    """Filter data with rolling mean with bfill. Automatically detect time column so we skip it."""
    processed_df = df.copy()
    
    # Process only non-time columns (skip the first column which is time)
    cols_to_process = df.columns[1:]
    
    # Apply rolling mean filtering with backfill to handle NaN values at the beginning
    for col in cols_to_process:
        processed_df[col] = df[col].rolling(window=window_size).mean().bfill()
    return processed_df

# Check and print missing values in each raw dataset
# for name, df in raw_dfs.items():
#     check_missing_values(name, df)

# Apply preprocessing to all datasets
processed_dfs = {}
for name, df in raw_dfs.items():
    processed_dfs[name] = preprocess_dataframe(df)

# Save processed data as CSV files
for name, df in processed_dfs.items():
    file_name = f"{name}_processed.csv"
    df.to_csv(os.path.join('processed', file_name), index=False)
    print(f"✅ Processed data saved: {file_name}")

# Store processed data in HDF5 file for easy access
with h5py.File("dataset.h5", "a") as f:
    for name, df in processed_dfs.items():
        person = name.split('_')[0].capitalize()
        activity = name.split('_')[1]
        f[f"Pre-processed Data/{person}/{activity}"] = df.values
#endregion

#region ---------------------------------PLOT ALL RAW DATA (X, Y, Z AXES)-----------------------------------------------
# Plot all raw data for each axis in 3x2 grid (3 users, walking vs jumping)
print("Plotting all raw data for each axis (X, Y, Z)...")

# Create figure with 3 rows (one for each person) and 2 columns (walking vs jumping)
fig_raw_all, axes_raw = plt.subplots(3, 2, figsize=(18, 12))
fig_raw_all.suptitle('Raw Acceleration Data - All Axes', fontsize=16)

# Common columns across all datasets
time_col = raw_dfs['lorenzo_walking'].columns[0]  # Time column
x_col = raw_dfs['lorenzo_walking'].columns[1]     # X-axis
y_col = raw_dfs['lorenzo_walking'].columns[2]     # Y-axis
z_col = raw_dfs['lorenzo_walking'].columns[3]     # Z-axis

# Row 0: Lorenzo data (walking vs jumping)
# Walking
axes_raw[0, 0].plot(raw_dfs['lorenzo_walking'][time_col], raw_dfs['lorenzo_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[0, 0].plot(raw_dfs['lorenzo_walking'][time_col], raw_dfs['lorenzo_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[0, 0].plot(raw_dfs['lorenzo_walking'][time_col], raw_dfs['lorenzo_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[0, 0].set_title('Lorenzo Walking (Raw)')
axes_raw[0, 0].set_xlabel(time_col)
axes_raw[0, 0].set_ylabel('Acceleration (m/s²)')
axes_raw[0, 0].grid(True)
axes_raw[0, 0].legend()

# Jumping
axes_raw[0, 1].plot(raw_dfs['lorenzo_jumping'][time_col], raw_dfs['lorenzo_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[0, 1].plot(raw_dfs['lorenzo_jumping'][time_col], raw_dfs['lorenzo_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[0, 1].plot(raw_dfs['lorenzo_jumping'][time_col], raw_dfs['lorenzo_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[0, 1].set_title('Lorenzo Jumping (Raw)')
axes_raw[0, 1].set_xlabel(time_col)
axes_raw[0, 1].set_ylabel('Acceleration (m/s²)')
axes_raw[0, 1].grid(True)
axes_raw[0, 1].legend()

# Row 1: KayKay data (walking vs jumping)
# Walking
axes_raw[1, 0].plot(raw_dfs['kaykay_walking'][time_col], raw_dfs['kaykay_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[1, 0].plot(raw_dfs['kaykay_walking'][time_col], raw_dfs['kaykay_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[1, 0].plot(raw_dfs['kaykay_walking'][time_col], raw_dfs['kaykay_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[1, 0].set_title('KayKay Walking (Raw)')
axes_raw[1, 0].set_xlabel(time_col)
axes_raw[1, 0].set_ylabel('Acceleration (m/s²)')
axes_raw[1, 0].grid(True)
axes_raw[1, 0].legend()

# Jumping
axes_raw[1, 1].plot(raw_dfs['kaykay_jumping'][time_col], raw_dfs['kaykay_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[1, 1].plot(raw_dfs['kaykay_jumping'][time_col], raw_dfs['kaykay_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[1, 1].plot(raw_dfs['kaykay_jumping'][time_col], raw_dfs['kaykay_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[1, 1].set_title('KayKay Jumping (Raw)')
axes_raw[1, 1].set_xlabel(time_col)
axes_raw[1, 1].set_ylabel('Acceleration (m/s²)')
axes_raw[1, 1].grid(True)
axes_raw[1, 1].legend()

# Row 2: Daniil data (walking vs jumping)
# Walking
axes_raw[2, 0].plot(raw_dfs['daniil_walking'][time_col], raw_dfs['daniil_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[2, 0].plot(raw_dfs['daniil_walking'][time_col], raw_dfs['daniil_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[2, 0].plot(raw_dfs['daniil_walking'][time_col], raw_dfs['daniil_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[2, 0].set_title('Daniil Walking (Raw)')
axes_raw[2, 0].set_xlabel(time_col)
axes_raw[2, 0].set_ylabel('Acceleration (m/s²)')
axes_raw[2, 0].grid(True)
axes_raw[2, 0].legend()

# Jumping
axes_raw[2, 1].plot(raw_dfs['daniil_jumping'][time_col], raw_dfs['daniil_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_raw[2, 1].plot(raw_dfs['daniil_jumping'][time_col], raw_dfs['daniil_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_raw[2, 1].plot(raw_dfs['daniil_jumping'][time_col], raw_dfs['daniil_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_raw[2, 1].set_title('Daniil Jumping (Raw)')
axes_raw[2, 1].set_xlabel(time_col)
axes_raw[2, 1].set_ylabel('Acceleration (m/s²)')
axes_raw[2, 1].grid(True)
axes_raw[2, 1].legend()

plt.tight_layout()
plt.show()
print("Raw data plots displayed.")

# Plot absolute acceleration for walking and jumping raw data
print("Plotting absolute acceleration for walking and jumping raw data...")

# Set dark theme for plots like in generate_images.py
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#212529',
    'axes.facecolor': '#212529',
    'savefig.facecolor': '#212529',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#495057',
    'axes.grid': True,
    'font.size': 12
})

# Side-by-side layout for raw data (walking left, jumping right)
fig_abs, axes_abs = plt.subplots(1, 2, figsize=(15, 6))
fig_abs.suptitle('Raw Absolute Acceleration Data', fontsize=16)

# Calculate absolute acceleration for walking (using KayKay's data)
if 'Absolute acceleration (m/s^2)' in raw_dfs['kaykay_walking'].columns:
    abs_acc_walking = raw_dfs['kaykay_walking']['Absolute acceleration (m/s^2)'].values
else:
    # Calculate absolute acceleration from x, y, z components
    x = raw_dfs['kaykay_walking']['Linear Acceleration x (m/s^2)'].values
    y = raw_dfs['kaykay_walking']['Linear Acceleration y (m/s^2)'].values
    z = raw_dfs['kaykay_walking']['Linear Acceleration z (m/s^2)'].values
    abs_acc_walking = np.sqrt(x**2 + y**2 + z**2)  # Proper absolute value calculation

# Calculate absolute acceleration for jumping (using Lorenzo's data)
if 'Absolute acceleration (m/s^2)' in raw_dfs['lorenzo_jumping'].columns:
    abs_acc_jumping = raw_dfs['lorenzo_jumping']['Absolute acceleration (m/s^2)'].values
else:
    # Calculate absolute acceleration from x, y, z components
    x = raw_dfs['lorenzo_jumping']['Linear Acceleration x (m/s^2)'].values
    y = raw_dfs['lorenzo_jumping']['Linear Acceleration y (m/s^2)'].values
    z = raw_dfs['lorenzo_jumping']['Linear Acceleration z (m/s^2)'].values
    abs_acc_jumping = np.sqrt(x**2 + y**2 + z**2)  # Proper absolute value calculation

# Plot walking absolute acceleration (left)
axes_abs[0].plot(raw_dfs['kaykay_walking'][time_col], abs_acc_walking, 'g-', linewidth=2, alpha=0.9, label='Absolute Acceleration')
axes_abs[0].set_title('Walking Activity (Raw Absolute Acceleration)', fontsize=14)
axes_abs[0].set_xlabel('Time (s)')
axes_abs[0].set_ylabel('Absolute Acceleration (m/s²)')
axes_abs[0].set_ylim(bottom=0)  # Ensure y-axis starts at 0
axes_abs[0].grid(True)
axes_abs[0].legend()

# Plot jumping absolute acceleration (right)
axes_abs[1].plot(raw_dfs['lorenzo_jumping'][time_col], abs_acc_jumping, 'r-', linewidth=2, alpha=0.9, label='Absolute Acceleration')
axes_abs[1].set_title('Jumping Activity (Raw Absolute Acceleration)', fontsize=14)
axes_abs[1].set_xlabel('Time (s)')
axes_abs[1].set_ylabel('Absolute Acceleration (m/s²)')
axes_abs[1].set_ylim(bottom=0)  # Ensure y-axis starts at 0
axes_abs[1].grid(True)
axes_abs[1].legend()

plt.tight_layout()
plt.show()
print("Absolute acceleration plots displayed.")

# Plot comparison of raw vs processed absolute acceleration data in a new window
print("Plotting raw vs processed absolute acceleration data...")
fig_proc, axes_proc = plt.subplots(1, 2, figsize=(15, 6))
fig_proc.suptitle('Raw vs Processed Absolute Acceleration Data', fontsize=16)

# Calculate absolute acceleration for walking (raw and processed)
if 'Absolute acceleration (m/s^2)' in raw_dfs['kaykay_walking'].columns:
    raw_abs_walking = raw_dfs['kaykay_walking']['Absolute acceleration (m/s^2)'].values
    processed_abs_walking = processed_dfs['kaykay_walking']['Absolute acceleration (m/s^2)'].values
else:
    # Calculate absolute acceleration from x, y, z components - raw data
    x_raw = raw_dfs['kaykay_walking']['Linear Acceleration x (m/s^2)'].values
    y_raw = raw_dfs['kaykay_walking']['Linear Acceleration y (m/s^2)'].values
    z_raw = raw_dfs['kaykay_walking']['Linear Acceleration z (m/s^2)'].values
    raw_abs_walking = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
    
    # Calculate absolute acceleration from x, y, z components - processed data
    x_proc = processed_dfs['kaykay_walking']['Linear Acceleration x (m/s^2)'].values
    y_proc = processed_dfs['kaykay_walking']['Linear Acceleration y (m/s^2)'].values
    z_proc = processed_dfs['kaykay_walking']['Linear Acceleration z (m/s^2)'].values
    processed_abs_walking = np.sqrt(x_proc**2 + y_proc**2 + z_proc**2)

# Calculate absolute acceleration for jumping (raw and processed)
if 'Absolute acceleration (m/s^2)' in raw_dfs['lorenzo_jumping'].columns:
    raw_abs_jumping = raw_dfs['lorenzo_jumping']['Absolute acceleration (m/s^2)'].values
    processed_abs_jumping = processed_dfs['lorenzo_jumping']['Absolute acceleration (m/s^2)'].values
else:
    # Calculate absolute acceleration from x, y, z components - raw data
    x_raw = raw_dfs['lorenzo_jumping']['Linear Acceleration x (m/s^2)'].values
    y_raw = raw_dfs['lorenzo_jumping']['Linear Acceleration y (m/s^2)'].values
    z_raw = raw_dfs['lorenzo_jumping']['Linear Acceleration z (m/s^2)'].values
    raw_abs_jumping = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
    
    # Calculate absolute acceleration from x, y, z components - processed data
    x_proc = processed_dfs['lorenzo_jumping']['Linear Acceleration x (m/s^2)'].values
    y_proc = processed_dfs['lorenzo_jumping']['Linear Acceleration y (m/s^2)'].values
    z_proc = processed_dfs['lorenzo_jumping']['Linear Acceleration z (m/s^2)'].values
    processed_abs_jumping = np.sqrt(x_proc**2 + y_proc**2 + z_proc**2)

# Plot walking raw vs processed absolute acceleration (left)
axes_proc[0].plot(raw_dfs['kaykay_walking'][time_col], raw_abs_walking, 'k-', alpha=0.5, label='Raw Absolute Acceleration')
axes_proc[0].plot(processed_dfs['kaykay_walking'][time_col], processed_abs_walking, 'b-', linewidth=2, alpha=0.8, label='Filtered Absolute Acceleration')
axes_proc[0].set_title('Walking: Raw vs Filtered (Absolute Acceleration)', fontsize=14)
axes_proc[0].set_xlabel('Time (s)')
axes_proc[0].set_ylabel('Absolute Acceleration (m/s²)')
axes_proc[0].set_ylim(bottom=0)  # Ensure y-axis starts at 0
axes_proc[0].grid(True)
axes_proc[0].legend()

# Plot jumping raw vs processed absolute acceleration (right)
axes_proc[1].plot(raw_dfs['lorenzo_jumping'][time_col], raw_abs_jumping, 'k-', alpha=0.5, label='Raw Absolute Acceleration')
axes_proc[1].plot(processed_dfs['lorenzo_jumping'][time_col], processed_abs_jumping, 'r-', linewidth=2, alpha=0.8, label='Filtered Absolute Acceleration')
axes_proc[1].set_title('Jumping: Raw vs Filtered (Absolute Acceleration)', fontsize=14)
axes_proc[1].set_xlabel('Time (s)')
axes_proc[1].set_ylabel('Absolute Acceleration (m/s²)')
axes_proc[1].set_ylim(bottom=0)  # Ensure y-axis starts at 0
axes_proc[1].grid(True)
axes_proc[1].legend()

plt.tight_layout()
plt.show()
print("Raw vs processed absolute acceleration plots displayed.")

#region ---------------------------------PLOT ALL PROCESSED DATA (X, Y, Z AXES)-----------------------------------------------
# Plot all processed data for each axis in 3x2 grid (3 users, walking vs jumping)
print("Plotting all processed data for each axis (X, Y, Z)...")

# Create figure with 3 rows (one for each person) and 2 columns (walking vs jumping)
fig_processed_all, axes_processed = plt.subplots(3, 2, figsize=(18, 12))
fig_processed_all.suptitle('Processed Acceleration Data - All Axes', fontsize=16)

# Row 0: Lorenzo data (walking vs jumping)
# Walking
axes_processed[0, 0].plot(processed_dfs['lorenzo_walking'][time_col], processed_dfs['lorenzo_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[0, 0].plot(processed_dfs['lorenzo_walking'][time_col], processed_dfs['lorenzo_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[0, 0].plot(processed_dfs['lorenzo_walking'][time_col], processed_dfs['lorenzo_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[0, 0].set_title('Lorenzo Walking (Processed)')
axes_processed[0, 0].set_xlabel(time_col)
axes_processed[0, 0].set_ylabel('Acceleration (m/s²)')
axes_processed[0, 0].grid(True)
axes_processed[0, 0].legend()

# Jumping
axes_processed[0, 1].plot(processed_dfs['lorenzo_jumping'][time_col], processed_dfs['lorenzo_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[0, 1].plot(processed_dfs['lorenzo_jumping'][time_col], processed_dfs['lorenzo_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[0, 1].plot(processed_dfs['lorenzo_jumping'][time_col], processed_dfs['lorenzo_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[0, 1].set_title('Lorenzo Jumping (Processed)')
axes_processed[0, 1].set_xlabel(time_col)
axes_processed[0, 1].set_ylabel('Acceleration (m/s²)')
axes_processed[0, 1].grid(True)
axes_processed[0, 1].legend()

# Row 1: KayKay data (walking vs jumping)
# Walking
axes_processed[1, 0].plot(processed_dfs['kaykay_walking'][time_col], processed_dfs['kaykay_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[1, 0].plot(processed_dfs['kaykay_walking'][time_col], processed_dfs['kaykay_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[1, 0].plot(processed_dfs['kaykay_walking'][time_col], processed_dfs['kaykay_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[1, 0].set_title('KayKay Walking (Processed)')
axes_processed[1, 0].set_xlabel(time_col)
axes_processed[1, 0].set_ylabel('Acceleration (m/s²)')
axes_processed[1, 0].grid(True)
axes_processed[1, 0].legend()

# Jumping
axes_processed[1, 1].plot(processed_dfs['kaykay_jumping'][time_col], processed_dfs['kaykay_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[1, 1].plot(processed_dfs['kaykay_jumping'][time_col], processed_dfs['kaykay_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[1, 1].plot(processed_dfs['kaykay_jumping'][time_col], processed_dfs['kaykay_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[1, 1].set_title('KayKay Jumping (Processed)')
axes_processed[1, 1].set_xlabel(time_col)
axes_processed[1, 1].set_ylabel('Acceleration (m/s²)')
axes_processed[1, 1].grid(True)
axes_processed[1, 1].legend()

# Row 2: Daniil data (walking vs jumping)
# Walking
axes_processed[2, 0].plot(processed_dfs['daniil_walking'][time_col], processed_dfs['daniil_walking'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[2, 0].plot(processed_dfs['daniil_walking'][time_col], processed_dfs['daniil_walking'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[2, 0].plot(processed_dfs['daniil_walking'][time_col], processed_dfs['daniil_walking'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[2, 0].set_title('Daniil Walking (Processed)')
axes_processed[2, 0].set_xlabel(time_col)
axes_processed[2, 0].set_ylabel('Acceleration (m/s²)')
axes_processed[2, 0].grid(True)
axes_processed[2, 0].legend()

# Jumping
axes_processed[2, 1].plot(processed_dfs['daniil_jumping'][time_col], processed_dfs['daniil_jumping'][x_col], 'r-', alpha=0.7, label='X-axis')
axes_processed[2, 1].plot(processed_dfs['daniil_jumping'][time_col], processed_dfs['daniil_jumping'][y_col], 'g-', alpha=0.7, label='Y-axis')
axes_processed[2, 1].plot(processed_dfs['daniil_jumping'][time_col], processed_dfs['daniil_jumping'][z_col], 'b-', alpha=0.7, label='Z-axis')
axes_processed[2, 1].set_title('Daniil Jumping (Processed)')
axes_processed[2, 1].set_xlabel(time_col)
axes_processed[2, 1].set_ylabel('Acceleration (m/s²)')
axes_processed[2, 1].grid(True)
axes_processed[2, 1].legend()

plt.tight_layout()
plt.show()
print("Processed data plots displayed.")

#region ---------------------------------PLOT DATA (RAW vs FILTERED)-----------------------------------------------
fig_lorenzo, axes = plt.subplots(3, 2, figsize=(15, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
fig_lorenzo.suptitle('Acceleration Processing', fontsize=16)
print("Plotting Raw vs filtered absolute acceleration for all users....")

# Lorenzo Walking
time_col = raw_dfs['lorenzo_walking'].columns[0]
abs_col  = raw_dfs['lorenzo_walking'].columns[3]

ax1.plot(raw_dfs['lorenzo_walking'][time_col], raw_dfs['lorenzo_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax1.plot(processed_dfs['lorenzo_walking'][time_col], processed_dfs['lorenzo_walking'][abs_col],'r-', label='Filtered Abs-acceleration')

ax1.set_title('Lorenzo Walking')
ax1.set_xlabel(time_col)
ax1.set_ylabel(abs_col)
ax1.grid(True)
ax1.legend()

# Lorenzo Jumping
time_col = raw_dfs['lorenzo_jumping'].columns[0]
abs_col  = raw_dfs['lorenzo_jumping'].columns[3]

ax2.plot(raw_dfs['lorenzo_jumping'][time_col], raw_dfs['lorenzo_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax2.plot(processed_dfs['lorenzo_jumping'][time_col], processed_dfs['lorenzo_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax2.set_title('Lorenzo Jumping')
ax2.set_xlabel(time_col)
ax2.set_ylabel(abs_col)
ax2.grid(True)
ax2.legend()

# Kaykay Walking
time_col = raw_dfs['kaykay_walking'].columns[0]
abs_col  = raw_dfs['kaykay_walking'].columns[3]

ax3.plot(raw_dfs['kaykay_walking'][time_col], raw_dfs['kaykay_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax3.plot(processed_dfs['kaykay_walking'][time_col], processed_dfs['kaykay_walking'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax3.set_title('KayKay Walking')
ax3.set_xlabel(time_col)
ax3.set_ylabel(abs_col)
ax3.grid(True)
ax3.legend()

# Kaykay Jumping
time_col = raw_dfs['kaykay_jumping'].columns[0]
abs_col  = raw_dfs['kaykay_jumping'].columns[3]

ax4.plot(raw_dfs['kaykay_jumping'][time_col], raw_dfs['kaykay_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax4.plot(processed_dfs['kaykay_jumping'][time_col], processed_dfs['kaykay_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax4.set_title('KayKay Jumping')
ax4.set_xlabel(time_col)
ax4.set_ylabel(abs_col)
ax4.grid(True)
ax4.legend()

# Daniil Walking
time_col = raw_dfs['daniil_walking'].columns[0]
abs_col  = raw_dfs['daniil_walking'].columns[3]

ax5.plot(raw_dfs['daniil_walking'][time_col], raw_dfs['daniil_walking'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')
ax5.plot(processed_dfs['daniil_walking'][time_col], processed_dfs['daniil_walking'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax5.set_title('Daniil Walking')
ax5.set_xlabel(time_col)
ax5.set_ylabel(abs_col)
ax5.grid(True)
ax5.legend()

# Daniil Jumping
time_col = raw_dfs['daniil_jumping'].columns[0]
abs_col  = raw_dfs['daniil_jumping'].columns[3]

ax6.plot(raw_dfs['daniil_jumping'][time_col], raw_dfs['daniil_jumping'][abs_col], 'k-', alpha=0.7, label='Raw Abs-acceleration')    
ax6.plot(processed_dfs['daniil_jumping'][time_col], processed_dfs['daniil_jumping'][abs_col], 'r-', label='Filtered Abs-acceleration')

ax6.set_title('Daniil Jumping')
ax6.set_xlabel(time_col)
ax6.set_ylabel(abs_col)
ax6.grid(True)
ax6.legend()
plt.tight_layout()
plt.show()
print("Raw vs filtered absolute acceleration for all users plotted.")

#plot histogram lorenzo walking acceleration
x_data = raw_dfs['lorenzo_walking']["Linear Acceleration x (m/s^2)"]
y_data = raw_dfs['lorenzo_walking']["Linear Acceleration y (m/s^2)"]
z_data = raw_dfs['lorenzo_walking']["Linear Acceleration z (m/s^2)"]
plt.figure(figsize=(12, 7))

# Plot all four histograms on the same axis
print("Plotting histogram of Lorenzo Walking - All Acceleration Axes")
plt.hist(x_data, bins=30, alpha=0.5, label='X Acceleration', edgecolor='black')
plt.hist(y_data, bins=30, alpha=0.5, label='Y Acceleration', edgecolor='black')
plt.hist(z_data, bins=30, alpha=0.5, label='Z Acceleration', edgecolor='black')

# Titles and labels
plt.title('Histogram of Lorenzo Walking - All Acceleration Axes')
plt.xlabel('Acceleration (m/s²)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#plot histogram lorenzo jumping acceleration
print("Plotting histogram of Lorenzo Jumping - All Acceleration Axes")
x_data = raw_dfs['lorenzo_jumping']["Linear Acceleration x (m/s^2)"]
y_data = raw_dfs['lorenzo_jumping']["Linear Acceleration y (m/s^2)"]
z_data = raw_dfs['lorenzo_jumping']["Linear Acceleration z (m/s^2)"]
plt.figure(figsize=(12, 7))

# Plot all four histograms on the same axis
plt.hist(x_data, bins=30, alpha=0.5, label='X Acceleration', edgecolor='black')
plt.hist(y_data, bins=30, alpha=0.5, label='Y Acceleration', edgecolor='black')
plt.hist(z_data, bins=30, alpha=0.5, label='Z Acceleration', edgecolor='black')

# Titles and labels
plt.title('Histogram of Lorenzo Jumping - All Acceleration Axes')
plt.xlabel('Acceleration (m/s²)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#endregion

#region ---------------------------------SEGMENT DATA-----------------------------------------------
def segment_data_5s(data, window_size):
    """Splits data into segments of length window_size, skipping the time column."""
    segments = []

    # Identify time col to skip it
    time_col = data.columns[0]
    
    # Use everything except time col
    data_no_time = data.drop(columns=[time_col])

    # Create segments of fixed window size from the data
    for i in range(0, len(data_no_time), window_size):
        segment = data_no_time.iloc[i : i + window_size, :].values
        # Only add complete segments (avoids partial segments at the end)
        if len(segment) == window_size:
            segments.append(segment)
    return np.array(segments)

# Compute average sampling frequency for each person to determine appropriate window size
lorenzo_sampling_rate = 1 / processed_dfs['lorenzo_walking'][time_col].diff().mean()
kaykay_sampling_rate = 1 / processed_dfs['kaykay_jumping'][time_col].diff().mean()
daniil_sampling_rate = 1 / processed_dfs['daniil_jumping'][time_col].diff().mean()

print(f"Lorenzo Estimated Sampling Frequency: {lorenzo_sampling_rate:.2f} Hz")
print(f"KayKay Estimated Sampling Frequency: {kaykay_sampling_rate:.2f} Hz")
print(f"Daniil Estimated Sampling Frequency: {daniil_sampling_rate:.2f} Hz")

# Round sampling rates to nearest 50 Hz and check if they are consistent
if (round(lorenzo_sampling_rate/50)*50) == (round(kaykay_sampling_rate/50)*50) == (round(daniil_sampling_rate/50)*50):
    official_sample_rate = (round(lorenzo_sampling_rate/50)*50)
else:
    print("Warning: Sampling rates are not the same across datasets")

print(f"Official Sample rate: {official_sample_rate}")

# Calculate window size for 5-second segments based on sampling rate
window_size = int(5 * official_sample_rate)

# Segment all processed datasets
segmented_arrays = {}
for name, df in processed_dfs.items():
    segmented_arrays[name] = segment_data_5s(df, window_size)

# Print the shape of each segmented array for verification
print("Segment shapes:")
for name, array in segmented_arrays.items():
    print(f"{name}: {array.shape}")
#endregion

#region ---------------------------------EXTRACT FEATURES-----------------------------------------------
def extract_features(segment):
    '''Extract statistical features from a segment of data.'''
    features = {}
    # Basic statistical features
    features["mean"] = np.mean(segment, axis=0)  # Mean of each axis
    features["std"] = np.std(segment, axis=0)    # Standard deviation
    features["min"] = np.min(segment, axis=0)    # Minimum value
    features["max"] = np.max(segment, axis=0)    # Maximum value
    features["range"] = np.ptp(segment, axis=0)  # Peak-to-peak (max-min)
    features["variance"] = np.var(segment, axis=0)  # Variance
    features["median"] = np.median(segment, axis=0)  # Median value
    features["rms"] = np.sqrt(np.mean(np.square(segment), axis=0))  # Root mean square
    
    # Shape-based features
    features["kurtosis"] = stats.kurtosis(segment, axis=0)  # Kurtosis (peakedness)
    features["skewness"] = stats.skew(segment, axis=0)      # Skewness (asymmetry)
    return features

# List of all feature names
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'variance',
    'median', 'rms', 'kurtosis', 'skewness'
]

def features_to_dataframe(features_list):
    '''Convert features list to a dataframe with named columns.'''
    # Axes names for column labeling
    axes = ['x', 'y', 'z', 'abs']
    
    # Create column names by combining feature name and axis
    columns = []
    for feature in feature_names:
        for axis in axes:
            columns.append(f"{feature}_{axis}")
    
    # Convert the feature dictionaries to rows of data
    data_rows = []
    for feat_dict in features_list:
        row = []
        for feature in feature_names:
            row.extend(feat_dict[feature])
        data_rows.append(row)
    
    return pd.DataFrame(data_rows, columns=columns)

# Extract features from each segment of data
features_arrays = {}
for name, array in segmented_arrays.items():
    feats_for_name = []
    for segment in array:
        feats_for_name.append(extract_features(segment))
    features_arrays[name] = feats_for_name

# Convert feature lists to DataFrames with named columns
features_dfs = {}
for name, feat_list in features_arrays.items():
    features_dfs[name] = features_to_dataframe(feat_list)

# Save the feature DataFrames to CSV files
for name, df in features_dfs.items():
    out_file = os.path.join('segmented', f'{name}_segmented.csv')
    df.to_csv(out_file, index=False)
    print(f"✅ Segmented data saved: {out_file}")
#endregion

#region ---------------------------------CREATE FINAL DATASET-----------------------------------------------
# Combine all feature DataFrames into a single dataset, adding activity labels
final_dataset = pd.DataFrame()
for name, df in features_dfs.items():
    # Label: 0 for walking, 1 for jumping
    activity_label = 0 if name.endswith('walking') else 1  
    df['activity'] = activity_label
    final_dataset = pd.concat([final_dataset, df], axis=0, ignore_index=True)

print(f"Final dataset shape: {final_dataset.shape}")  # (rows, columns)
#endregion

#region ---------------------------------SPECIFY WHICH AXIS TO TRAIN WITH-----------------------------------------------
# Organize columns by axis type to help with feature selection and analysis

# Get all feature columns (exclude the activity/label column)
feature_columns = final_dataset.columns[:-1]
num_axes = 4  # x, y, z, abs

# Use modulus to extract features for each axis separately
# This works because our columns are organized such that:
# Column 0, 4, 8... are x-axis features
# Column 1, 5, 9... are y-axis features
# Column 2, 6, 10... are z-axis features
# Column 3, 7, 11... are abs-axis features
x_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 0]
y_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 1]
z_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 2]
abs_cols = [col for i, col in enumerate(feature_columns) if i % num_axes == 3]

# Print feature lists for verification
print("X features:", x_cols)
print("Y features:", y_cols)
print("Z features:", z_cols)
print("Abs features:", abs_cols)

# Reorganize final dataset to group features by axis type
final_dataset = final_dataset[x_cols + y_cols + z_cols + abs_cols + ['activity']]
#endregion

#region ---------------------------------TRAIN LOGISTIC REGRESSION-----------------------------------------------
# Prepare data for model training by separating features and labels
final_data = final_dataset.drop('activity', axis=1)
final_labels = final_dataset['activity']

# Split data into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(
    final_data, final_labels, test_size=0.1, shuffle=True, random_state=0
)

# Store train/test data in HDF5 file
with h5py.File("dataset.h5", "a") as f:
    f["Segmented Data/Train/X"] = X_train.values
    f["Segmented Data/Train/y"] = y_train.values
    
    f["Segmented Data/Test/X"] = X_test.values
    f["Segmented Data/Test/y"] = y_test.values
    
    # Store feature names and count for reference
    f["Segmented Data/feature_names"] = list(final_data.columns)
    f["Segmented Data/num_features"] = len(final_data.columns)

# Print dataset information
print(f"Number of features (columns) being used: {len(final_data.columns)}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create and train a logistic regression model with preprocessing pipeline
# The pipeline ensures standardization of features before model training
l_reg = LogisticRegression(max_iter=10000)  # Increase max_iter to ensure convergence
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(X_train, y_train)

# Save the trained model to disk for later use
joblib.dump(clf, "activity_classifier.pkl")

# Evaluate model performance on test data
predictions = clf.predict(X_test)
clf_probs = clf.predict_proba(X_test)  # Probability estimates
acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)

# Print evaluation metrics
print(f"Accuracy: {acc}")
print(f"Recall: {recall}")

# Calculate feature correlation with activity label for feature importance analysis
correlation = final_dataset.corr()['activity'].sort_values(ascending=False)
print("Correlation of final dataset:")
print(correlation)
#endregion

#region ---------------------------------PLOT CORRELATION BAR GRAPH-----------------------------------------------
# Visualize correlation between each feature and the activity label
print("Plotting feature importance correlation...")
fig_correlation, ax = plt.subplots(figsize=(12, 8))
fig_correlation.suptitle('Top 10 Features by Correlation with Activity', fontsize=16)

# Sort correlation values and get top 10 features
sorted_corr = correlation.sort_values(ascending=False)
top_10_corr = sorted_corr.head(10)  # Get only top 10 features

# Create horizontal bar chart
bars = ax.barh(top_10_corr.index, top_10_corr.values, color='#20c997', alpha=0.8)

# Add correlation values next to bars
for i, (feature, corr) in enumerate(top_10_corr.items()):
    ax.text(corr + 0.01, i, f'{corr:.2f}', va='center', fontsize=10, color='white')

# Set axis labels and limits
ax.set_xlabel('Correlation', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_xlim(0, 1.1)  # Set x-axis limit for horizontal bar chart

# Invert y-axis to show highest correlation at the top
ax.invert_yaxis()

# Format y-axis labels
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)

# Add grid
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
print("Feature importance correlation plot displayed.")
#endregion

# Save the final model again to ensure it's stored
joblib.dump(clf, 'activity_classifier.pkl')
print("✅Model saved as activity_classifier.pkl")
