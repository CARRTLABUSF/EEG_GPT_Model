import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -------------------------------
# Define custom metric
# -------------------------------
def angle_accuracy(threshold=5):
    def custom_accuracy(y_true, y_pred):
        diff = K.abs(y_true - y_pred)
        correct = K.less_equal(diff, threshold)
        return K.mean(K.cast(correct, dtype='float32'))
    custom_accuracy.__name__ = "custom_accuracy"
    return custom_accuracy

# -------------------------------
# Load model and data
# -------------------------------
print("📦 Loading model, test data, and scaler...")

model = load_model(
    "best_joint_angle_model.h5",
    custom_objects={
        'mean_squared_error': MeanSquaredError(),
        'custom_accuracy': angle_accuracy(threshold=5)
    }
)

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
scalers = joblib.load("scalers.pkl")
scaler_eeg = scalers["eeg"]
scaler_mocap = scalers["mocap"]


print("✅ Model loaded successfully.")

# -------------------------------
# Predict & Unscale
# -------------------------------
print("🔮 Making predictions...")
y_pred = model.predict(X_test)

y_pred_unscaled = scaler_mocap.inverse_transform(y_pred)
y_test_unscaled = scaler_mocap.inverse_transform(y_test)

# -------------------------------
# Overall Evaluation
# -------------------------------
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
r2 = r2_score(y_test_unscaled, y_pred_unscaled)

print(f"\n📊 Overall Evaluation Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# -------------------------------
# Limb-Specific Evaluation
# -------------------------------
angle_labels = [
    'Shoulder_X', 'Shoulder_Y', 'Shoulder_Z',
    'Elbow_X',
    'Wrist_X', 'Wrist_Y', 'Wrist_Z'
]

limb_indices = {
    'Shoulder': [0, 1, 2],
    'Elbow': [3],
    'Wrist': [4, 5, 6]
}

print("\n📊 Limb-Specific Evaluation Metrics:")
for limb, indices in limb_indices.items():
    y_true_limb = y_test_unscaled[:, indices]
    y_pred_limb = y_pred_unscaled[:, indices]

    limb_mae = mean_absolute_error(y_true_limb, y_pred_limb)
    limb_mse = mean_squared_error(y_true_limb, y_pred_limb)
    limb_r2 = r2_score(y_true_limb, y_pred_limb)

    print(f"\n🦾 {limb} Metrics:")
    print(f"MAE: {limb_mae:.4f}")
    print(f"MSE: {limb_mse:.4f}")
    print(f"R² Score: {limb_r2:.4f}")
