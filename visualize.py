import numpy as np
import matplotlib.pyplot as plt

# Load dataset
X = np.load("dataset_20240709_X_3096.npy")
y = np.load("dataset_20240709_y_3096.npy")

# Convert one-hot → labels
if len(y.shape) > 1:
    y = np.argmax(y, axis=1)

unique_labels = np.unique(y)
current_index = 0

def show_gesture(label_index):
    label = unique_labels[label_index]
    
    # pick one sample of this label
    idx = np.where(y == label)[0][0]
    sample = X[idx]

    frame = sample[-1]  # last frame

    left_hand = frame[:63].reshape(21, 3)
    right_hand = frame[63:].reshape(21, 3)

    plt.clf()

    # plot points
    plt.scatter(left_hand[:,0], left_hand[:,1], c='blue', label='Left')
    plt.scatter(right_hand[:,0], right_hand[:,1], c='red', label='Right')

    plt.title(f"Gesture Label: {int(label)}")
    plt.gca().invert_yaxis()
    plt.legend()

    plt.draw()

def on_key(event):
    global current_index

    if event.key == 'n':  # next
        current_index = (current_index + 1) % len(unique_labels)
        show_gesture(current_index)

    elif event.key == 'p':  # previous
        current_index = (current_index - 1) % len(unique_labels)
        show_gesture(current_index)

    elif event.key == 'q':  # quit
        plt.close()

# Setup figure
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)

show_gesture(current_index)
plt.show()