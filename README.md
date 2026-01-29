# ML Activity Classifier

A machine learning project that classifies human activities (walking vs. jumping) using smartphone accelerometer data. Developed as part of ELEC 292: Introduction to Data Science.

## 📋 Project Overview

This project uses supervised machine learning techniques to classify physical activities based on linear acceleration data collected from smartphones. The system processes raw accelerometer readings, extracts statistical features, and trains a logistic regression model to distinguish between walking and jumping activities.

## ✨ Features

- **Data Collection**: Accelerometer data collection using the Phyphox mobile app
- **Preprocessing**: Rolling mean filter to reduce noise in raw sensor data
- **Segmentation**: Time-series data segmented into 5-second windows
- **Feature Extraction**: 10 statistical features extracted per axis (mean, std, min, max, range, variance, median, RMS, kurtosis, skewness)
- **Machine Learning**: Logistic regression classifier with standardized features
- **Interactive GUI**: Desktop application for real-time activity classification
- **Real-time Classification**: Live activity detection via Phyphox web interface (bonus feature)

## 🛠️ Technologies Used

- **Python 
- **Data Processing**: pandas, numpy, h5py
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib
- **Statistical Analysis**: scipy
- **GUI Development**: tkinter
- **Web Automation**: selenium (for bonus feature)

## 📁 Project Structure

```
ML_activity_classifier-main/
├── model_training.py          # Main script for data processing and model training
├── gui_app.py                 # Desktop GUI application for activity classification
├── bonus.py                   # Real-time classification via Phyphox web interface
├── hdf5.py                    # HDF5 file utilities
├── activity_classifier.pkl    # Trained logistic regression model
├── dataset.h5                 # Organized HDF5 dataset (raw, processed, segmented)
├── raw/                       # Raw accelerometer data CSV files
├── processed/                 # Preprocessed data with noise filtering
└── segmented/                 # Segmented data in 5-second windows
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy h5py matplotlib scipy scikit-learn joblib
```

For the GUI application:
```bash
pip install tkinter
```

For the bonus real-time feature:
```bash
pip install selenium
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML_activity_classifier.git
cd ML_activity_classifier
```

2. Ensure you have the required data files in the `raw/` directory

3. Run the model training script:
```bash
python model_training.py
```

### Usage

#### Training the Model
Run the complete data pipeline and train the classifier:
```bash
python model_training.py
```

This script will:
- Load raw accelerometer data
- Apply preprocessing (rolling mean filter)
- Segment data into 5-second windows
- Extract statistical features
- Train a logistic regression model
- Generate visualization plots
- Save the trained model as `activity_classifier.pkl`

#### Using the GUI Application
Launch the desktop application for classifying new accelerometer data:
```bash
python gui_app.py
```

Features:
- Load CSV files containing accelerometer data
- Visualize raw acceleration data
- Real-time activity classification
- Display prediction confidence scores

#### Real-time Classification (Bonus)
Use live smartphone data via Phyphox:
```bash
python bonus.py
```

Make sure Phyphox app is running and accessible via the network.

## 📊 Data Processing Pipeline

1. **Raw Data Collection**: Accelerometer data (x, y, z axes + absolute acceleration) collected at ~100 Hz
2. **Preprocessing**: 51-sample rolling mean filter to smooth sensor noise
3. **Segmentation**: Data divided into 5-second windows (500 samples per segment)
4. **Feature Extraction**: 40 features total (10 statistical features × 4 axes)
5. **Model Training**: Logistic regression with standard scaling
6. **Evaluation**: Model performance assessed using accuracy and recall metrics

## 🎯 Model Performance

The trained logistic regression classifier achieves:
- High accuracy in distinguishing walking from jumping activities
- Robust performance across different individuals
- Real-time classification capability

Key discriminative features include acceleration range, standard deviation, and RMS values.

## 👥 Contributors

This project was completed as a pair programming assignment for ELEC 292: Introduction to Data Science.

- Daniil Nistribenko
- LorenzoDeMarni

## 📝 Course Information

**Course**: ELEC 292 - Introduction to Data Science  
**Institution**: Queen's University

## 🙏 Acknowledgments

- **Phyphox**: Mobile app for data collection
- **scikit-learn**: Machine learning framework
- Course instructors and teaching assistants

## 📄 License

This project is created for educational purposes as part of a university course assignment.

## 🔮 Future Improvements

- Add more activity classes (running, climbing stairs, etc.)
- Implement deep learning models (LSTM, CNN)
- Develop mobile application for on-device classification
- Add feature selection to optimize model performance
- Collect larger, more diverse datasets
