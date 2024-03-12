import tensorflow as tf
import numpy as np

# Examples
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

# Labels
y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)


# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.relu))  # Use ReLU activation
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


model.summary()


history = model.fit(x, y, batch_size=1, epochs=500)


predictions = model.predict_on_batch(x)
with open('my_output.txt', 'w', encoding='utf-8') as f:
    f.write(predictions)  # Replace with how you're saving the output
