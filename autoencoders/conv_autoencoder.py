"""Testing convolutional autoencoder with MNIST."""

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

import matplotlib.pyplot as plt

# %% loading data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x = x_train.reshape((60000, 28, 28, 1))

# %% model architecture
inputs = Input(shape=(28, 28, 1))

h = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
h = MaxPooling2D((2, 2))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
# h = MaxPooling2D((2, 2))(h)
# h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
encoded = MaxPooling2D((2, 2), padding='same')(h)

h = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
h = UpSampling2D((2, 2))(h)
h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
#h = UpSampling2D((2, 2))(h)
#h = Conv2D(16, (3, 3), activation='relu')(h)
h = UpSampling2D((2, 2))(h)

outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)

# %% Compile model

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')

# %% train model
model.fit(x, x, batch_size=64, epochs=3, verbose=2, validation_split=0.05,
          )

# %% predict for new data
y = x_test[15, :, :].reshape((1, 28, 28, 1))
pred = model.predict(y)

plt.imshow(y.reshape((28, 28)))
plt.show()

plt.imshow(pred.reshape((28, 28)))
plt.show()
