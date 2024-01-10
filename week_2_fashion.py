import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras


print(tf.__version__)

#Load the fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

#calling load_data() on this object will give you two tuples with the two lists each.
# These will be training and testing values for the graphics that contain the clothing items
# and their labels.

# Load the training and test split of the fashion MNIST dataset 
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

#Lets prints a training model
index = 0 

#Set number of characters per orw when printing
np.set_printoptions(linewidth=320)

#Print the label and image
print (f'LABEL:{training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
plt.imshow(training_images[index])
