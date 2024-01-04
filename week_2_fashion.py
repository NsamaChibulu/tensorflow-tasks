import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

#Load the fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

#calling load_data() on this object will give you two tuples with the two lists each.
# These will be training and testing values for the graphics that contain the clothing items
# and their labels.

# Load the training and test split of the fashion MNIST dataset 
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()