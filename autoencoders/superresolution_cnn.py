"""Super resolution CNN"""

from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

SIGMA = 3
MIN_RES = (4, 4)

BATCH_SIZE = 32

# %% loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_norm = np.expand_dims(x_train, axis=3) / 255
x_test_norm = np.expand_dims(x_test, axis=3) / 255

# %% add noise to data
x_train_blur = x_train_norm.copy()
x_test_blur = x_test_norm.copy()

print("Preprocessing training data...")
for i in range(x_train_norm.shape[0]):
    print(i)
    x_train_blur[i] = \
        np.expand_dims(
            cv2.resize(
                cv2.resize(gaussian_filter(x_train_norm[i].squeeze(),
                                           sigma=SIGMA),
                           dsize=MIN_RES),
                dsize=(28, 28), interpolation=cv2.INTER_NEAREST
            ),
            axis=3
        )

print("Preprocessing testing data...")
for i in range(x_test_norm.shape[0]):
    print(i)
    x_test_blur[i] = \
        np.expand_dims(
            cv2.resize(
                cv2.resize(gaussian_filter(x_test_norm[i].squeeze(),
                                           sigma=SIGMA),
                           dsize=MIN_RES),
                dsize=(28, 28), interpolation=cv2.INTER_NEAREST
            ),
            axis=3
        )

# %% validate preprocessed data
plt.imshow(np.concatenate([x_test_norm[1].squeeze(),
                           x_test_blur[1].squeeze()],
                          axis=1))
plt.show()

# %% model architecture
inputs = Input(shape=(28, 28, 1))

h = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(4, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(4, (3, 3), activation='relu', padding='same')(h)

outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

# %% Compile model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=0.0001), loss='mse')
print(model.summary())

# %% train model
callbacks = [
    TensorBoard(log_dir=f'./logs/{datetime.now()}', batch_size=BATCH_SIZE,
                write_images=True)
]

model.fit(x_train_blur, x_train_norm, batch_size=BATCH_SIZE,
          epochs=40, verbose=2, validation_split=0.02, callbacks=callbacks)

# %% predict for new data
x_pred = model.predict(x_test_blur).squeeze()

# %% visualize example
for i in range(4):
    img_concat = np.concatenate([x_test_norm.squeeze()[i],
                                 x_test_blur.squeeze()[i],
                                 x_pred[i]], axis=1)
    plt.imshow(img_concat)
    plt.show()
