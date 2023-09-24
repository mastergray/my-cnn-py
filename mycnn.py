import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

class MyCNN:

    # CONSTURCTOR :: NUBER, NUMBER, NUMBER, NUMBER -> self
    def __init__(self, width, height, channels, num_classes):
        self.width = width                   # Width of image we are training on (in pixels)
        self.height = height                 # Height of image we are training on (in pixels)
        self.channels = channels             # Number of color channels for the image, so 1 for grey scale and 3 for RGB
        self.num_classes = num_classes       # This is the number of categories we are trying to classify an image in
        self.model = self.createModel()     # This actually implements the model we are trying to train

    # :: VOID -> keras.Model 
    def createModel(self):
        
        # Defines input shape of input data:
        input_shape = (self.width, self.height, self.channels)

        # Defines what size and number of color channels of the images we are training on (e.g. (28, 28, 1) for MINST)
        inputs = keras.Input(shape=input_shape)

        # Applies convolution with a "sliding window" size of 3x3 using 32 filters to learn features from input data
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)

        # Downsamples the input data using a 2x2 window by selecting the maximum value from the values present in that 2x2 region, where this maximum value represents the most prominent feature or activation within that local region:
        x = layers.MaxPooling2D(pool_size=2)(x)

         # Applies convolution with a "sliding window" size of 3x3 using 64 filters to learn features from input data
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)

        # Downsamples the input data again using 2x2 - this allows for "focusing" on more prominent features over a larger feature map - given that the previous layer has increased it's filters from 32 to 64:
        x = layers.MaxPooling2D(pool_size=2)(x)

         # Applies convolution with a "sliding window" size of 3x3 using 128 filters to learn features from input data
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)

        # Flattens our 3D outputs to 1D for the output layer:
        x = layers.Flatten()(x)

        # We use "softmax" for the output layer activaton function because it converts the network's raw output scores into a probability distribution over multiple classes:
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # Initialzie the model we will train: 
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile model with optimize, loss function, and metrics:
        model.compile(optimizer="rmsprop",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        
        return model

    # :: NUMPY -> NUMPY
    # This will normalize and reshape sample data so it be used by the model:
    def prepareSamples(self, samples):
        samples = samples.reshape((len(samples), self.width, self.height, self.channels))
        samples = samples.astype("float32") / 255
        return samples


    # :: NUMPY, NUMPY, NUMBER, NUMBER, NUMBER -> self
    # Set hyperparameters and trains model:
    # NOTE: Assumes training_samples have been prepared already!
    # NOTE: We try to set "seed" to always set model from same randomized weights when traing:
    def trainModel(self, training_samples, training_labels, epochs=5, batch_size=64, seed=123):
     
        # Set the seed for numpy
        np.random.seed(seed)

        # Set the seed for TensorFlow
        tf.random.set_seed(seed)

        # Set random seed for Python  
        random.seed(seed)

        # Train model:
        self.model.fit(training_samples, training_labels, epochs=epochs, batch_size=batch_size)
        return self
    

    # NUMPY, NUMPY -> self
    # Returns accurary of model using test data:
    # NOTE: Assumes testing_samples have been prepared already!
    def evalModel(self, test_samples, test_labels):
        test_loss, test_acc = self.model.evaluate(test_samples, test_labels)
        print(f"Test accuracy: {test_acc:.3f}")
        return self

    # :: STRING -> self
    # Save weights of trained model:
    def saveWeights(self, filename):
        self.model.save_weights(filename)
        return self


    # :: STRING -> self
    # Loads weights into model:
    def loadWeights(self, filename):
        self.model.load_weights(filename)
        return self
