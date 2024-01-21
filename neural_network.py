from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from kerastuner import HyperModel, Objective
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from cleaning import *
import matplotlib.pyplot as plt
from compute import annotated_bar_chart

# Model number 2: Neural Network

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(Dense(units=hp.Int('units_input',
                                     min_value=32,
                                     max_value=256,
                                     step=32),
                        activation='relu',
                        input_shape=self.input_shape))
        for i in range(hp.Int('n_layers', 1, 5)):
            model.add(Dense(units=hp.Int('units_hidden_' + str(i),
                                         min_value=32,
                                         max_value=256,
                                         step=32),
                            activation='relu'))
            model.add(Dropout(rate=hp.Float('dropout_' + str(i),
                                            min_value=0.0,
                                            max_value=0.5,
                                            step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',
                                                                values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error',
                      metrics=[RootMeanSquaredError()])
        return model


# Clean and prepare the data
df = pd.read_csv("data/training.csv")
main_data, colnames = remove_correlations(clean_data(), 0.9)
oil_prod = main_data.pop("OilPeakRate")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(main_data.to_numpy(), oil_prod.to_numpy(), test_size=0.2,
                                                    random_state=7413)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize the tuner
tuner = RandomSearch(
    MyHyperModel(input_shape=(X_train.shape[1],)),
    objective=Objective("val_root_mean_squared_error", direction="min"),
    max_trials=1,
    executions_per_trial=1,
    directory='my_dir',
    project_name='keras_tuner_oil_peak_rate'
)

# Define early stopping criteria
early_stopper = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopper])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model with the best hyperparameters
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopper])

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

# Generate predictions
predictions = model.predict(X_test).flatten()

# Select a random sample of 25 or 30 values from the test set and their corresponding predictions
sample_size = 25  # or 30 if you prefer
indices = np.random.choice(range(len(y_test)), sample_size, replace=False)
sampled_predictions = predictions[indices]
sampled_actuals = y_test[indices]

annotated_bar_chart(sample_size, sampled_actuals, sampled_predictions, indices)

ultimate_min = np.min(np.vstack((predictions, y_test)))
ultimate_max = np.max(np.vstack((predictions, y_test)))

plt.scatter(X_test[:, 0], X_test[:, 1], c = np.log(predictions + np.e), vmin=np.log(ultimate_min + np.e), vmax=np.log(ultimate_max + np.e))
plt.colorbar()
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c = np.log(y_test + np.e), vmin=np.log(ultimate_min + np.e), vmax=np.log(ultimate_max + np.e))
plt.colorbar()
plt.show()