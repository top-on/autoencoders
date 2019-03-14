"""Testing convolutional autoencoder with MNIST."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

# %% loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_norm = np.expand_dims(x_train, axis=3) / 255
x_test_norm = np.expand_dims(x_test, axis=3) / 255

# %% add noise to data
noise_rate = 0.05
x_train_noisy = x_train_norm + np.random.choice([0, 1],
                                                p=[1 - noise_rate, noise_rate],
                                                size=x_train_norm.shape)
x_test_noisy = x_test_norm + np.random.choice([0, 1],
                                              p=[1 - noise_rate, noise_rate],
                                              size=x_test_norm.shape)

x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

# %% model architecture
inputs = Input(shape=(28, 28, 1))

h = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
h = MaxPooling2D((2, 2))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D((2, 2))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
encoded = MaxPooling2D((2, 2), padding='same')(h)

h = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
h = UpSampling2D((2, 2))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = UpSampling2D((2, 2))(h)
h = Conv2D(16, (3, 3), activation='relu')(h)
h = UpSampling2D((2, 2))(h)

outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

# %% Compile model

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=0.001), loss='mse')
print(model.summary())

# %% train model

callbacks = [
    TensorBoard(log_dir='./logs/run6', batch_size=64, write_images=True)
]

model.fit(x_train_noisy, x_train_norm, batch_size=64, epochs=20, verbose=2,
          validation_split=0.02, callbacks=callbacks)

# %% predict for new data
x_pred = model.predict(x_test_noisy).squeeze()

# %% visualize example
i = 6
img_concat = np.concatenate([x_test_noisy.squeeze()[i], x_pred[i]], axis=1)
plt.imshow(img_concat)
plt.show()
