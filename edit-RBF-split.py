import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

from sklearn.model_selection import train_test_split

features = data.iloc[:, 5:105].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(features_scaled[:,1:3])
targets = data[['y1', 'y2', 'y3']].values

features_train_scaled, features_test_scaled, targets_train, targets_test = train_test_split(
    features_scaled, targets, test_size=0.3, random_state=42
)



# Check and handle NaN and infinite values
features_train_scaled = np.nan_to_num(features_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
features_test_scaled = np.nan_to_num(features_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
targets_train = np.nan_to_num(targets_train, nan=0.0, posinf=0.0, neginf=0.0)
targets_test = np.nan_to_num(targets_test, nan=0.0, posinf=0.0, neginf=0.0)

# Define the custom RBF layer
class RBF(tf.keras.layers.Layer):
    def __init__(self, num_centers, num_features, **kwargs):
        super(RBF, self).__init__(**kwargs)
        self.num_centers = num_centers
        self.num_features = num_features

    def build(self, input_shape):
        self.centers = self.add_weight(
            shape=(self.num_centers, self.num_features),
            initializer='random_normal',
            trainable=False
        )
        self.sigmas = self.add_weight(
            shape=(self.num_centers,),
            initializer='ones',
            trainable=False
        )
        super(RBF, self).build(input_shape)

    def call(self, inputs):
        dists = tf.reduce_sum(tf.square(tf.expand_dims(inputs, 1) - self.centers), axis=-1)
        return tf.exp(-dists / (2 * tf.square(self.sigmas)))

# Custom callback to save loss to a file
class RBF_LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('history/RBF_loss_history.txt', 'a') as f:
            f.write(f"{logs['loss']} {logs['val_loss']}\n")

# Build the RBF network model
num_centers = 100

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(features_train_scaled.shape[1],)),
    RBF(num_centers, features_train_scaled.shape[1]),
    tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.0),
    tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dropout(0.0),
    tf.keras.layers.Dense(3)
])

# Compile the model
initial_learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
loss_history = RBF_LossHistory()
model.fit(features_train_scaled, targets_train, epochs=20, batch_size=64, validation_split=0.1,
          callbacks=[early_stopping, loss_history])

# Test the model
y_test_pred = model.predict(features_test_scaled)
mse = mean_squared_error(targets_test, y_test_pred)
print(f'Test MSE: {mse}')

# Save predictions to an Excel file
predictions_df = pd.DataFrame(y_test_pred, columns=['y1', 'y2', 'y3'])
predictions_df.to_excel('predictions.xlsx', index=False)
