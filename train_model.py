import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Function to preprocess input data
def preprocess_input_data(input_data, scaler):
    input_data = np.array(input_data).reshape(-1, 1)
    normalized_input = scaler.transform(input_data).reshape(1, -1, 1)
    return normalized_input

# Function to make predictions using the loaded model
def make_predictions(model, input_data, scaler):
    normalized_input = preprocess_input_data(input_data, scaler)
    predictions = model.predict(normalized_input)
    
    # Post-process predictions
    predictions_scalar = (predictions.flatten() * 79) + 1
    clipped_predictions = np.clip(predictions_scalar, 1, 80)
    rounded_predictions = np.floor(clipped_predictions).astype(int)
    
    unique_predictions = list(set(rounded_predictions))
    
    return unique_predictions

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = [data[i: i + seq_length] for i in range(len(data) - seq_length)]
    return np.array(sequences)

# Load data
try:
    data = pd.read_csv('history.csv')
except FileNotFoundError as e:
    print(f"Error: 'history.csv' file not found. {e}")

# Ensure datetime format for the 'Date-Time (Input)' column
data['Date-Time (Input)'] = pd.to_datetime(data['Date-Time (Input)'], format='%Y-%d-%m %H:%M', errors='coerce')

# Extract and preprocess the 'Input' column with a fixed sequence length of 20 values
X = data[' Input'].apply(lambda x: list(map(int, x.split())))
max_seq_length = max(map(len, X))

# Pad sequences with zeros
X_padded = pad_sequences(X, maxlen=max_seq_length, padding='post', truncating='post', value=0)

# Assuming sum is the desired operation
y_normalized = (X_padded.sum(axis=1) - 1) / 79.0

seq_length = 20
X_sequences = create_sequences(X_padded, seq_length)
y_sequences = create_sequences(y_normalized, seq_length)
print("X_sequences shape:", X_sequences.shape)
print("y_sequences shape:", y_sequences.shape)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Reshape X_train and X_test
X_train = X_train.reshape((X_train.shape[0], seq_length, max_seq_length))
X_test = X_test.reshape((X_test.shape[0], seq_length, max_seq_length))
print("X_train shape before reshaping:", X_train.shape)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_normalized = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
print("X_train_normalized shape before reshaping:", X_train_normalized.shape)

# Build the LSTM model with adjusted layers
model = Sequential([
    LSTM(32, input_shape=(seq_length, max_seq_length), return_sequences=True, activation='tanh'),
    LSTM(16, activation='tanh', return_sequences=True),
    LSTM(8, activation='tanh'),
    Dense(1, activation='linear')
])

# Load the trained model and scaler
model_path = 'prediction_model_lstm'

# Load the MinMaxScaler
scaler = MinMaxScaler()
try:
    scaler_params = np.load('scaler_params.npy', allow_pickle=True)
    scaler_min, scaler_scale = scaler_params.item().get('data_min_'), scaler_params.item().get('data_max_')
except FileNotFoundError:
    print("Error: 'scaler_params.npy' file not found.")

# If the scaler_params.npy file is not found, fit and transform the data to obtain scaler parameters
if 'scaler_params' not in locals():
    X_train_sample = np.random.rand(100, 1)  # You can replace this with your actual training data
    scaler.fit(X_train_sample)

    # Save the MinMaxScaler parameters
    joblib.dump(scaler, 'scaler_params.joblib')

# Replace the use of tf.keras.optimizers.legacy.Adam with tf.keras.optimizers.Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Evaluate the model (optional)
score = model.evaluate(X_test_normalized, y_test)
print(f'Model Loss: {score}')

# Train the model with adjusted early stopping patience
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train_normalized, y_train, epochs=400, batch_size=32, validation_data=(X_test_normalized, y_test), callbacks=[early_stopping])

# Display the model summary
model.summary()

# Save the trained model with a version number
model.save('prediction_model_lstm_v1', save_format='tf')
