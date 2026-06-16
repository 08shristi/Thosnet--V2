import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import mediapipe as mp
from collections import deque

# Model
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

    out = layers.Dense(9, activation='softmax')(x)

    return tf.keras.Model([left_in, right_in], out)

model = build_model()
model.load_weights('models/thosnet_weights.h5')

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Buffers
buffer = deque(maxlen=30)
pred_buffer = deque(maxlen=10)

class_names = [f"Gesture_{i}" for i in range(9)]

# Camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("THOSnet Live", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    features = np.zeros((2,21,3))

    if results.multi_hand_landmarks:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            idx = 0 if handedness.classification[0].label == 'Left' else 1
            features[idx] = coords
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    else:
        buffer.clear()
        pred_buffer.clear()

    # Prepare sequence
    lh = features[0].reshape(-1)
    rh = features[1].reshape(-1)
    buffer.append(np.concatenate([lh, rh]))

    # Prediction
    if len(buffer) == 30:
        seq = np.array(buffer)[None, :, :]
        p = model.predict([seq[:, :, :63], seq[:, :, 63:]], verbose=0)

        pred = np.argmax(p)
        conf = np.max(p)

        pred_buffer.append(pred)

        if len(pred_buffer) == 10:
            final_pred = max(set(pred_buffer), key=pred_buffer.count)

            if conf > 0.7:
                label = class_names[final_pred]
            else:
                label = "No Gesture"
        else:
            label = "Stabilizing..."
    else:
        label = "Collecting..."

    # Display
    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("THOSnet Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()