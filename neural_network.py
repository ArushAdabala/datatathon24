from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from cleaning import clean_data
import matplotlib.pyplot as plt


# Clean and prepare the data
main_data = clean_data()
oil_prod = main_data.pop("OilPeakRate")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(main_data.to_numpy(), oil_prod.to_numpy(), test_size=0.5, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Sequential model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # Increased neurons
model.add(Dense(64, activation='relu'))  # Additional layer
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(Dense(64, activation='relu'))  # Additional layer
model.add(Dense(32, activation='relu'))  # Existing layer
model.add(Dropout(0.5))  # Another dropout layer
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])

"""
# Define early stopping criteria
early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the early stopping callback
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopper])
"""

history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test RMSE: {rmse:.4f}")

# Plotting the learning curves
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Plot training & validation RMSE values
plt.subplot(1, 2, 2)
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
plt.title('Root Mean Squared Error')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()

plt.show()
