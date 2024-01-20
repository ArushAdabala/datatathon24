from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from cleaning import clean_data

# Clean and prepare the data
main_data = clean_data()
oil_prod = main_data.pop("OilPeakRate")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(main_data.to_numpy(), oil_prod.to_numpy(), test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Sequential model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test RMSE: {rmse:.4f}")
