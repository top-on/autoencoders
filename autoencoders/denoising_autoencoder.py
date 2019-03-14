"""Testing convolutional autoencoder with MNIST."""

import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

# %% loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train_norm = np.expand_dims(x_train, axis=3) / 255

# %% model architecture
inputs = Input(shape=(28, 28, 1))

h = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
h = MaxPooling2D((4, 4))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D((2, 2))(h)
h = Conv2D(4, (3, 3), activation='relu', padding='same')(h)
encoded = MaxPooling2D((2, 2), padding='same')(h)

h = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
h = UpSampling2D((4, 4))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
h = UpSampling2D((2, 2))(h)
h = Conv2D(16, (3, 3), activation='relu')(h)
h = UpSampling2D((2, 2))(h)

outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

# %% Compile model

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
print(model.summary())

# %% train model

callbacks = [
    TensorBoard(log_dir='./logs/run6', batch_size=64, write_images=True)
]

model.fit(x_train_norm, x_train_norm, batch_size=64, epochs=20, verbose=2, validation_split=0.02,
          callbacks=callbacks)

# %% predict for new data
x_test_norm = x_test / 255
x_test_expand = np.expand_dims(x_test_norm, axis=3)
x_pred = model.predict(x_test_expand).squeeze()

x_diff = np.abs(x_test_norm - x_pred)

# %% visualize example
i = 8
plt.imshow(x_test_norm[i])
plt.show()
plt.imshow(x_pred[i])
plt.show()
plt.imshow(x_diff[i])
plt.show()

# %% difference per digit

# calculate norm of each difference, vectorized
x_diff_reshaped = x_diff.reshape(x_diff.shape[0], -1)
norms = np.linalg.norm(x_diff_reshaped, axis=1)

# plot distributions of norms
sns.boxplot(y_test, norms)
plt.show()
