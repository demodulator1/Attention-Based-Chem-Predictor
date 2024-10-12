import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model

# Load data
data = pd.read_csv('data.csv')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

from sklearn.model_selection import train_test_split

features = data.iloc[:, 5:105].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)



# print(features_scaled[:,1:3])
targets = data[['y1', 'y2', 'y3']].values
# scaler2 = StandardScaler()
# targets = scaler2.fit_transform(targets)


features_train_scaled, features_test_scaled, targets_train, targets_test = train_test_split(
    features_scaled, targets, test_size=0.3, random_state=42
)



# Check and handle NaN and infinite values
features_train_scaled = np.nan_to_num(features_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
features_test_scaled = np.nan_to_num(features_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
targets_train = np.nan_to_num(targets_train, nan=0.0, posinf=0.0, neginf=0.0)
targets_test = np.nan_to_num(targets_test, nan=0.0, posinf=0.0, neginf=0.0)

# Build BP neural network model with attention mechanism
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(units, activation='softmax')

    def call(self, inputs):
        attention_weights = self.dense(inputs)
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=1)
        return weighted_sum

# Define the model using Functional API
inputs = tf.keras.Input(shape=(features_train_scaled.shape[1],))
x = tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
x = tf.keras.layers.Dropout(0.0)(x)
x = tf.keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
x = tf.keras.layers.Dropout(0.0)(x)
x = tf.keras.layers.Reshape((1, 8))(x)
x = AttentionLayer(units=8)(x)
outputs = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Set learning rate to 0.001
initial_learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Custom callback to save loss to a file
class LossHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('history/Attention_loss_history.txt', 'a') as f:
            f.write(f"{logs['loss']} {logs['val_loss']}\n")

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
loss_history = LossHistory()
model.fit(features_train_scaled, targets_train, epochs=20, batch_size=64, validation_split=0.1,
          callbacks=[early_stopping, loss_history])

# Test the model
y_test_pred = model.predict(features_test_scaled)
mse = mean_squared_error(targets_test, y_test_pred)
print(f'Test MSE: {mse}')

# Save predictions to an Excel file
predictions_df = pd.DataFrame(y_test_pred, columns=['y1', 'y2', 'y3'])
predictions_df.to_excel('predictions.xlsx', index=False)
