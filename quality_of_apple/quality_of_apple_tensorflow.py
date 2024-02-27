import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

# Load the data
csv_file_path = r'C:\Users\new\Desktop\ODE ML\quality_of_apple\apple_quality.csv'
df = pd.read_csv(csv_file_path)

# Omit id column
data = df.drop(['A_id', 'Quality'], axis=1)

# Normalize the entire dataset
normalized_data = normalize(data)

# Split data into training and test sets
train_data, test_data, expected_output_for_train, expected_output_for_test = train_test_split(
    normalized_data, df['Quality'], test_size=0.33, random_state=5)

expected_output_as_integer_train = (expected_output_for_train == 'good').astype(int).to_numpy().reshape(-1, 1)

# Initiate the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(7, )),
    tf.keras.layers.Dense(15, activation='sigmoid'),
    tf.keras.layers.Dense(7, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', 'mse'])

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # or other metrics like 'val_accuracy'
    patience=20,          # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)

# The validation set is a set of data, separate from the training set, that is used to
# validate our model performance during training. And to use early_stopping
history = model.fit(
    train_data,
    expected_output_as_integer_train,
    batch_size=23,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stopping])


# Evaluate the model on the test set
predictions = model.predict(test_data)
conf_matrix = confusion_matrix((expected_output_for_test == 'good').astype(int), predictions > 0.5)
print("Confusion Matrix:\n", conf_matrix)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
