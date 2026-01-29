from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import joblib
import pandas as pd
import numpy as np
from scipy import stats

# PHYPOX server URL
PHYPOX_URL = "http://192.168.2.59/"

# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-infobars")

# Load the model
model = joblib.load("activity_classifier.pkl")

# Start browser
browser = webdriver.Chrome(options=chrome_options)
browser.get(PHYPOX_URL)

datacols = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]

dataset = pd.DataFrame(columns=datacols)

# Window size for segmenting data
window_size = 20

# Feature extraction function
def extract_features(segment):
    features = {}
    segment = segment.astype(float)  # Ensure numeric type
    # Basic features
    features["mean"] = np.mean(segment, axis=0)
    features["std"] = np.std(segment, axis=0)
    features["min"] = np.min(segment, axis=0)
    features["max"] = np.max(segment, axis=0)
    features["range"] = np.ptp(segment, axis=0)
    features["variance"] = np.var(segment, axis=0)
    features["median"] = np.median(segment, axis=0)
    features["rms"] = np.sqrt(np.mean(np.square(segment), axis=0))
    features["kurtosis"] = stats.kurtosis(segment, axis=0)
    features["skewness"] = stats.skew(segment, axis=0)

    return features

# Define feature names
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'variance',
    'median', 'rms', 'kurtosis', 'skewness'
]
def features_to_dataframe(features_list):
    axes = ['x', 'y', 'z', 'abs']
    columns = [f"{feature}_{axis}" for feature in feature_names for axis in axes]
    
    data = []
    for features in features_list:  
        row = [features[feature] for feature in feature_names]
        row = np.concatenate(row)  # Flatten array
        data.append(row)

    return pd.DataFrame(data, columns=columns)

def segment_data_5s(data, window_size):
    segments = []
    for i in range(0, len(data), window_size):
        segment = data.iloc[i:i + window_size, :].values
        if len(segment) == window_size:
            segments.append(segment)
    
    return np.array(segments)

# Main loop to collect data
while True:
    elements = WebDriverWait(browser, 5).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.valueNumber"))
    )

    values = []
    for element in elements:
        if element.text is not None:  # Fix for NoneType issue
            text = element.text.strip()
            if text:  # Ensure it's not empty
                try:
                    values.append(float(text))
                except ValueError:
                    print(f"Skipping invalid value: {text}")
                    continue
    
    # Ensure we get 4 valid values
    if len(values) == len(datacols):
        new_row = pd.DataFrame([values], columns=datacols)
        dataset = pd.concat([dataset, new_row], ignore_index=True)

    print(f"Dataset Size: {len(dataset)}")

    # Check if dataset reached the required window size
    if len(dataset) >= window_size:
        segments = segment_data_5s(dataset, window_size)
        if len(segments) == 0:
            print("Warning: No valid segments created!")
            continue
        
        features_list = [extract_features(seg) for seg in segments]
        if len(features_list) == 0:
            print("Warning: No features extracted!")
            continue
        
        #feature extraction
        feature_df=features_to_dataframe(features_list)
        # only use absolute acceleration features
        # Exclude the 'activity' column from feature extraction
        feature_columns = feature_df.columns[:]
        num_axes = 4  # x, y, z, abs

        # Use modulus to extract each axis's features
        x_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 0]
        y_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 1]
        z_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 2]
        abs_cols = [col for i, col in enumerate(feature_columns) if i % num_axes == 3]

        feature_df = feature_df[x_cols + y_cols + z_cols + abs_cols]
        if feature_df.empty:
            print("Warning: Features DataFrame is empty!")
            continue
        
        # Predict the activity
        predictions = model.predict(feature_df)
        labels = ["walking" if p == 0 else "jumping" for p in predictions]
        # Count occurrences
        jumping_count = (predictions == 1).sum()
        walking_count = (predictions == 0).sum()
        # Reset dataset
        dataset = pd.DataFrame(columns=datacols)
        # Create output DataFrame
        output_df = pd.DataFrame({'Segment': range(len(labels)), 'Prediction': labels})
        output_df.loc[len(output_df)] = ['Final Classification', 'Jumping' if jumping_count > walking_count else 'Walking']

        print("\n=== Classification Results ===")
        print(output_df)
        print("==============================\n")
        time.sleep(0.5)

    # time.sleep(0.1)  # Reduce CPU usage
