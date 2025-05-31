# This model uses 1D CNN and LSTM layers to process the EEG and MoCap data

# --------------------------------------------------------
# Step 0: Imports
# --------------------------------------------------------
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import joblib

print("🚀 Script started")
# --------------------------------------------------------
# Step 1: Load and Inspect Data
# --------------------------------------------------------
folder_map = {"Flexion": "FM", "Extension": "BM", "Abduction": "SM", "Random": "RM"}
file_paths = {}
movement_labels = {}

for label, folder in folder_map.items():
    full_folder_path = os.path.join(os.getcwd(), folder)
    for filename in os.listdir(full_folder_path):
        if filename.endswith(".csv"):
            movement_name = f"{label} - {filename}"
            file_path = os.path.join(folder, filename)
            file_paths[movement_name] = file_path
            movement_labels[movement_name] = label

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}
print(f"✅ Loaded {len(datasets)} trials.")

eeg_channels = ['AF3', 'T7', 'Pz', 'T8', 'AF4']
mocap_features = [
    'RShoulderAngles_X', 'RShoulderAngles_Y', 'RShoulderAngles_Z', 'RElbowAngles_X',
    'RWristAngles_X', 'RWristAngles_Y', 'RWristAngles_Z'
]

all_features = eeg_channels + mocap_features

# --------------------------------------------------------
# Step 2: EEG Cleaning
# --------------------------------------------------------

def angle_accuracy(threshold=10):
    def custom_accuracy(y_true, y_pred):
        diff = K.abs(y_true - y_pred)
        correct = K.less_equal(diff, threshold)
        return K.mean(K.cast(correct, dtype='float32'))
    return custom_accuracy

def butter_bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=120.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

fs = 120.0
z_thresh = 5
filtered_datasets = {}

for name, df in datasets.items():
    df_filtered = df.copy()
    for ch in eeg_channels:
        filtered = butter_bandpass_filter(df[ch].values, fs=fs)
        z_scores = (filtered - np.mean(filtered)) / np.std(filtered)
        filtered[np.abs(z_scores) > z_thresh] = np.nan
        df_filtered[ch] = filtered
    filtered_datasets[name] = df_filtered

# --------------------------------------------------------
# Step 3: Format Data
# --------------------------------------------------------
window_size = 120  # Number of samples per window
stride = 16
X, y = [], []

for name, df in filtered_datasets.items():
    label = movement_labels[name]
    data = df[all_features].dropna().values
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start:start + window_size]
        if window.shape[0] == window_size:
            X.append(window)
            y.append(label)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --------------------------------------------------------
# Step 3.1: Normalize EEG and MoCap Separately
# --------------------------------------------------------
X_eeg = X[:, :, :5]
X_mocap = X[:, :, 5:]

X_eeg_reshaped = X_eeg.reshape(-1, X_eeg.shape[-1])
X_mocap_reshaped = X_mocap.reshape(-1, X_mocap.shape[-1])

scaler_eeg = StandardScaler()
scaler_mocap = StandardScaler()

X_eeg_scaled = scaler_eeg.fit_transform(X_eeg_reshaped).reshape(X_eeg.shape)
X_mocap_scaled = scaler_mocap.fit_transform(X_mocap_reshaped).reshape(X_mocap.shape)

X_final = np.concatenate([X_eeg_scaled, X_mocap_scaled], axis=2)



# --------------------------------------------------------
# Step 4: CNN + LSTM Model
# --------------------------------------------------------
X_reg = X_final[:, :, :5]
y_reg = X_final[:, -1, 5:]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)


def build_baseline_model(input_shape, output_dim):
    model = Sequential([
        Input(shape=input_shape),  # This is all you need
        Conv1D(32, kernel_size=3, activation='relu'),  # No need to repeat input_shape here
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', angle_accuracy(threshold=3)])
    return model


model = build_baseline_model(input_shape=X_train.shape[1:], output_dim=y_train.shape[1])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=70,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
    
)


 # Step 4.1: Save Model and Scaler for Future Use 

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
joblib.dump(scaler_mocap, "scaler_mocap.pkl")
model.save("best_joint_angle_model.h5")
print("✅ Trained model saved as best_joint_angle_model.h5")

# --------------------------------------------------------
# Step 5: Prediction and Inverse Normalization
# --------------------------------------------------------
y_pred = model.predict(X_test)
y_pred_unscaled = scaler_mocap.inverse_transform(y_pred)
y_test_unscaled = scaler_mocap.inverse_transform(y_test)

# --------------------------------------------------------
# Step 6: Evaluation
# --------------------------------------------------------

final_train_accuracy = history.history['custom_accuracy'][-1] * 100
final_val_accuracy = history.history['val_custom_accuracy'][-1] * 100

print(f"\n✅ Final Train Accuracy (within ±5°): {final_train_accuracy:.2f}%")
print(f"✅ Final Val Accuracy (within ±5°): {final_val_accuracy:.2f}%")

mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print(f"\n📊 Evaluation Metrics (Original Scale):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# --------------------------------------------------------
# Step 7: Plots
# --------------------------------------------------------
# 📉 Loss Plot

epochs_trained = len(history.history['loss'])

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label=f'Train Loss (MSE) (Window={window_size})Epochs = {epochs_trained}')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

angle_labels = [
    'Shoulder_X', 'Shoulder_Y', 'Shoulder_Z',
    'Elbow_X', 
    'Wrist_X', 'Wrist_Y', 'Wrist_Z']
        
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

limb_indices = {
    'Shoulder': range(0, 3),
    'Elbow': [3],
    'Wrist': range(4,7)
}

for ax, (limb, indices) in zip(axs, limb_indices.items()):
    for i in indices:
        ax.plot(y_test_unscaled[:50, i], label=f'True {angle_labels[i]} (Window={window_size}) Epochs = {epochs_trained}')
        ax.plot(y_pred_unscaled[:50, i], '--', label=f'Pred {angle_labels[i]} (Window={window_size})')
    ax.set_title(f"{limb} Joint Angles")
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True)
    ax.legend()


plt.figure(figsize=(10, 4))
plt.plot(history.history['custom_accuracy'], label=f'Train Accuracy (±5°) (Window={window_size}) Epochs = {epochs_trained}')
plt.plot(history.history['val_custom_accuracy'], label=f'Val Accuracy (±5°) (Window={window_size}) Epochs = {epochs_trained}')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Proportion within ±5°")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------------
# --------------------------------------------------------
# Step 8: ML-Based Direction Classifier from Predicted Joint Angles
# --------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 8.1: Create labels from known motions
direction_map = {
    "Flexion": "Forward",
    "Extension": "Backward",
    "Abduction": "Sideways"
}

X_dir = []
y_dir = []

for name, df in filtered_datasets.items():
    label = movement_labels[name]
    if label not in direction_map:
        continue
    direction = direction_map[label]
    data = df[all_features].dropna().values
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start:start + window_size]
        if window.shape[0] == window_size:
            joint_vector = window[-1, 5:]  # MoCap only
            X_dir.append(joint_vector)
            y_dir.append(direction)

X_dir = np.array(X_dir)
y_dir = np.array(y_dir)

# Normalize to match the prediction space
X_dir_scaled = scaler_mocap.transform(X_dir)

# Step 8.2: Train direction classifier
X_train_dir, X_test_dir, y_train_dir, y_test_dir = train_test_split(X_dir_scaled, y_dir, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_dir, y_train_dir)

# Step 8.3: Classify predicted joint angles from Random motions
random_mask = np.array([label == "Random" for label in y])
random_X = X_reg[random_mask]
random_y_pred = model.predict(random_X)
random_y_pred_unscaled = scaler_mocap.inverse_transform(random_y_pred)

predicted_directions = clf.predict(scaler_mocap.transform(random_y_pred_unscaled))

# Print and visualize
print("\n🧠 ML-Classified Directions for Random Motion Windows:")
for i, label in enumerate(predicted_directions[:20]):
    print(f"Window {i}: {label}")

# Step 8.4: Confusion matrix on test classifier
y_pred_test = clf.predict(X_test_dir)
ConfusionMatrixDisplay.from_predictions(y_test_dir, y_pred_test, labels=["Forward", "Backward", "Sideways"])
plt.title(
    "Direction Classifier (Random Forest) - Validation Confusion Matrix",
    fontsize=16,    # make the title bigger
    pad=12          # optional: give it a bit more breathing room
)
plt.xlabel("Predicted label", fontsize=14)
plt.ylabel("True label", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# Optional: Save to file
joblib.dump(clf, "motion_direction_rf.pkl") # Save Motion Direction_rf
with open("motion_labels_ml.txt", "w") as f:
    for i, label in enumerate(predicted_directions):
        f.write(f"Window {i}: {label}\n")
print("✅ Saved motion directions to 'motion_labels_ml.txt'")
