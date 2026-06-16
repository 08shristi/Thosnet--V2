import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# =========================
# LOAD DATA
# =========================
X = np.load("dataset_20240709_X_3096.npy")
y = np.load("dataset_20240709_y_3096.npy")

print("X shape:", X.shape)   # (3096,30,126)
print("y shape:", y.shape)   # (3096,9)

#  DO NOT reshape y

# =========================
# SPLIT HANDS
# =========================
X_left  = X[:, :, :63]
X_right = X[:, :, 63:]

# =========================
# MODEL
# =========================
def build_model():
    left_in  = tf.keras.Input(shape=(30,63))
    right_in = tf.keras.Input(shape=(30,63))

    L = layers.Bidirectional(layers.GRU(128, return_sequences=True))(left_in)
    R = layers.Bidirectional(layers.GRU(128, return_sequences=True))(right_in)

    merged = layers.Concatenate()([L, R])
    flat = layers.Flatten()(merged)

    x = layers.Dense(128, activation='relu')(flat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # ✅ 9 classes
    out = layers.Dense(9, activation='softmax')(x)

    return tf.keras.Model([left_in, right_in], out)


model = build_model()

# =========================
# COMPILE (IMPORTANT)
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',   #  for one-hot labels
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================
import time

start_time = time.time()

history = model.fit(
    [X_left, X_right],
    y,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

end_time = time.time()

total_time = end_time - start_time
print("Total Training Time:", total_time, "seconds")

# =========================
# SAVE
# =========================
model.save_weights("models/thosnet_weights.h5")

print("Training completed ✅")
import matplotlib.pyplot as plt

# Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()

plt.savefig("accuracy_graph.png")
plt.show()


# Training Time Comparison
models = ['Old Model', 'THOSNet-X']
training_time = [7200, total_time]

plt.figure()
plt.bar(models, training_time)

plt.xlabel('Models')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')

plt.grid(axis='y')

plt.savefig("time_comparison.png")
plt.show()