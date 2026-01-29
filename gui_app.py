import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Text, Scrollbar, RIGHT, Y, END, X
import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats
from tkinter import Frame, Canvas, BOTH, LEFT, RIGHT, VERTICAL
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch  # ADD THIS to your imports at the top
from sklearn.preprocessing import StandardScaler
# load the trained model
model = joblib.load("activity_classifier.pkl")

#parameters-window size
window_size=500

#feature extraction
def extract_features(segment):
    features = {}
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

# define feature names
feature_names = [
    'mean', 'std', 'min', 'max', 'range', 'variance',
    'median', 'rms', 'kurtosis', 'skewness'
]


def features_to_dataframe(features_list):
    # axes definition
    axes = ['x', 'y', 'z', 'abs']

    # columns
    columns = []
    for feature in feature_names:
        for axis in axes:
            columns.append(f"{feature}_{axis}")

    # create the data rows
    data = []  # list of lists(rows)
    for features in features_list:  
        row = []
        for feature in feature_names:  
            row.extend(features[feature])
        data.append(row)

    #create and return data frame
    df = pd.DataFrame(data, columns=columns)
    return df

#function segmentation
def segment_data_5s(data, window_size):
    segments = []
    
    # Attempt to identify time column (usually first column)
    if "Time" in data.columns or "time" in data.columns:
        time_col = next(col for col in data.columns if "time" in col.lower())
        data_no_time = data.drop(columns=[time_col])
    else:
        # Default: assume first column is time
        data_no_time = data.iloc[:, 1:]
    
    for i in range(0, len(data_no_time), window_size):
        segment = data_no_time.iloc[i:i + window_size, :].values
        if len(segment) == window_size:
            segments.append(segment)

    return np.array(segments)

#prediction
def popup(output_df):
    popupwindow=Toplevel(root)
    popupwindow.title("Predictions")
    popupwindow.geometry("400x400")
    popupwindow.configure(bg="#2c3e50")

    text=Text(popupwindow,wrap="word",bg="#2c3e50",fg="#fdf6e3")
    text.pack(side=LEFT,fill=BOTH,expand=1)

    scrollbar = Scrollbar(popupwindow, command=text.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    text.configure(yscrollcommand=scrollbar.set)

    for i, row in output_df.iterrows():
        text.insert(END, f"Segment {row['Segment']}: {row['Prediction']}\n")

def save_classifications_to_csv(original_data, output_df, file_path):
    try:
        # Create a new DataFrame with just the segment data
        segments_df = output_df.copy()
        
        # Remove the summary row if it exists
        segments_df = segments_df[segments_df['Segment'] != 'Majority Classification']
        
        # Create a simplified DataFrame with only the required columns
        result_df = pd.DataFrame({
            'Segment_Number': segments_df['Segment'].astype(int),
            'Time_Range': segments_df['Time Range'],
            'Activity': segments_df['Prediction']
        })
        
        # Add overall classification to success message
        majority_class = output_df.loc[output_df['Segment'] == 'Majority Classification', 'Prediction'].values[0]
        
        # Save to CSV
        result_df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Classification data saved to {file_path}\nOverall activity: {majority_class}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save CSV file: {str(e)}")

def open_csv():
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not csv_file_path:
        return  # If no file is selected, exit the function

    try:
        # Read the selected CSV
        df = pd.read_csv(csv_file_path)

        # Segment data
        segments = segment_data_5s(df, window_size)
        if len(segments) == 0:
            messagebox.showerror("Error", "The selected CSV file does not contain enough data for segmentation.")
            return

        # Extract features
        features_list = []
        for seg in segments:
            features = extract_features(seg)
            features_list.append(features)

        if not features_list:
            messagebox.showerror("Error", "No features were extracted. Ensure the CSV has valid acceleration data.")
            return

        # Convert to DataFrame
        feature_df = features_to_dataframe(features_list)
        # Exclude the 'activity' column from feature extraction
        feature_columns = feature_df.columns[:]
        num_axes = 4  # x, y, z, abs

        # Use modulus to extract each axis's features
        x_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 0]
        y_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 1]
        z_cols   = [col for i, col in enumerate(feature_columns) if i % num_axes == 2]
        abs_cols = [col for i, col in enumerate(feature_columns) if i % num_axes == 3]
        # print(f"X acceleration features being used: {x_cols}")
        # print(f"Y acceleration features being used: {y_cols}")
        # print(f"Z acceleration features being used: {z_cols}")
        # print(f"Absolute acceleration features being used: {abs_cols}")
        
        feature_df = feature_df[x_cols + y_cols + z_cols + abs_cols]
        # print(feature_df)

        # Make predictions
        predictions = model.predict(feature_df)
        labels = ["walking" if p == 0 else "jumping" for p in predictions]

        # Count classifications
        jumping_count = (predictions == 1).sum()
        walking_count = (predictions == 0).sum()

        # Calculate time ranges for each segment
        time_ranges = []
        for i in range(len(labels)):
            start_time = i * 5  # Each segment is 5 seconds
            end_time = (i + 1) * 5
            time_ranges.append(f"{start_time}-{end_time}s")

        # Create output DataFrame
        output_df = pd.DataFrame({
            'Segment': range(len(labels)),
            'Time Range': time_ranges,
            'Prediction': labels
        })
        output_df["Segment"] = output_df["Segment"].astype(str)  # Convert to string for proper display
        output_df.loc[len(output_df)] = ['Majority Classification', 'Total Time', 'Jumping' if jumping_count > walking_count else 'Walking']
        plot_raw_data_with_classification(df, labels)

        # Update predictions display
        predictions_text.config(state=tk.NORMAL)  # Ensure it's editable
        predictions_text.delete(1.0, END)  # Clear previous predictions
        for i, row in output_df.iterrows():
            if row['Segment'] == 'Majority Classification':
                predictions_text.insert(END, f"{row['Segment']}: {row['Prediction']}\n")
            else:
                predictions_text.insert(END, f"Segment {row['Segment']} ({row['Time Range']}): {row['Prediction']}\n")
        predictions_text.config(state=tk.DISABLED)  # Disable editing after updating
        
        # Store data for later use by the save button
        global current_data, current_output
        current_data = df
        current_output = output_df
        
        # Enable the save button now that we have data to save
        save_button.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {str(e)}")

def save_results():
    if 'current_data' not in globals() or 'current_output' not in globals():
        messagebox.showwarning("Warning", "Please classify a CSV file first.")
        return
        
    # Ask user where to save the file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Classification Results"
    )
    
    if not file_path:
        return  # User cancelled the dialog
        
    # Save the data
    save_classifications_to_csv(current_data, current_output, file_path)

def plot_raw_data_with_classification(df, labels):
    try:
        # Compute absolute acceleration
        df['abs'] = df['Absolute acceleration (m/s^2)']

        # Clear any existing plot in the plot_frame
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # Create the figure - adjust height to ensure full visibility
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(df['abs'].values, label='Absolute Acceleration', color='black')

        # Highlight each segment
        for i, label in enumerate(labels):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            color = '#3498db' if label == 'walking' else '#e74c3c'
            ax.axvspan(start_idx, end_idx, facecolor=color, alpha=0.3)

        ax.set_title("Raw Absolute Acceleration with Activity Highlighted")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Absolute Acceleration")

        # Add legend for the colors
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Walking'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Jumping'),
            Patch(facecolor='black', label='Absolute Acceleration', linewidth=2)
        ]
        ax.legend(handles=legend_elements)

        # Pack the plot frame first to ensure it's visible in the layout
        plot_frame.pack(fill=BOTH, expand=True, pady=(0, 20))
        
        # Create a canvas frame to hold the plot with proper scrolling
        canvas_frame = Frame(plot_frame, bg="#1a1f2c")
        canvas_frame.pack(fill=BOTH, expand=True)
        
        # Display the plot
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Add a toolbar for navigation (optional)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Plot Error", f"Could not plot data: {str(e)}")

#GUI
root=tk.Tk()
root.configure(bg="#1a1f2c")  # Dark blue background
root.geometry("1200x900")  # Slightly wider
root.resizable(True, True)  # Make window resizable to allow user adjustments
root.title("Activity Classifier App")

# Initialize globals for storing current data
current_data = None
current_output = None

# Create a master frame to hold everything
master_frame = Frame(root, bg="#1a1f2c")
master_frame.pack(fill=BOTH, expand=1, padx=20, pady=20)

# Header
header_frame = Frame(master_frame, bg="#1a1f2c")
header_frame.pack(fill=X, pady=(0, 20))

label=tk.Label(header_frame, text="WELCOME TO THE ACTIVITY CLASSIFIER APP", font=('Comic Sans MS',33,'bold'), bg="#1a1f2c", fg="#ffffff")
label.pack(pady=10)

# Button frame for organizing buttons
button_frame = Frame(master_frame, bg="#1a1f2c")
button_frame.pack(fill=X, pady=10)

# Center the buttons
button_container = Frame(button_frame, bg="#1a1f2c")
button_container.pack(anchor="center")

button=tk.Button(button_container, text="SELECT CSV FILE TO BE CLASSIFIED", font=('Comic Sans MS',20), command=open_csv, fg="#ffffff", bg="#2c3e50", 
               activebackground="#34495e", activeforeground="white", highlightthickness=4,
               highlightbackground="#3498db", highlightcolor="#3498db")
button.pack(side=LEFT, padx=10)

# Add save button (initially disabled until data is loaded)
save_button = tk.Button(button_container, text="SAVE RESULTS TO CSV", font=('Comic Sans MS', 20), 
                       command=save_results, fg="#ffffff", bg="#2c3e50", state=tk.DISABLED,
                       activebackground="#34495e", activeforeground="white", highlightthickness=4,
                       highlightbackground="#2ecc71", highlightcolor="#2ecc71")
save_button.pack(side=LEFT, padx=10)

# Add text area for predictions
predictions_frame = Frame(master_frame, bg="#1a1f2c")
predictions_frame.pack(fill=X, pady=20)

predictions_label = tk.Label(predictions_frame, text="Classification Results:", font=('Comic Sans MS', 20, 'bold'), bg="#1a1f2c", fg="#ffffff")
predictions_label.pack(pady=10)

text_container = Frame(predictions_frame, bg="#1a1f2c")
text_container.pack(fill=X)

predictions_text = Text(text_container, wrap="word", font=('Comic Sans MS', 12), bg="#2c3e50", fg="#ffffff", height=12)
predictions_text.pack(side=LEFT, fill=BOTH, expand=True)

predictions_scrollbar = Scrollbar(text_container, command=predictions_text.yview)
predictions_scrollbar.pack(side=RIGHT, fill=Y)
predictions_text.configure(yscrollcommand=predictions_scrollbar.set)

# Add a frame for the plot underneath the results
plot_frame = Frame(master_frame, bg="#1a1f2c")
# Note: plot_frame is not packed immediately, it's packed when the plot is created

root.mainloop()