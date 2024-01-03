import tensorflow as tf 
import numpy as np 
from tensorflow import keras

print(tf.__version__)

#Build a simple sequential model. It has 1 layer with 1 neuron
# and the input shape to it is just 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

#Next up , you will declare inputs and outputs for training
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#Train the model
model.fit(xs, ys, epochs=1000)

#Make a predictions

print(model.predict([10.0]))