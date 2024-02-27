import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

expected_outputs = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),  # Try this with SGD and see the difference!
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy', 'mse', 'binary_accuracy'])
model.summary()


history = model.fit(inputs, expected_outputs, batch_size=1, epochs=500)


predictions = model.predict_on_batch(inputs)
print(predictions)

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


