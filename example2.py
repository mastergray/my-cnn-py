from mycnn import MyCNN 
from tensorflow.keras.datasets import mnist

# Load training and test data:
(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()

# Initalize the class to train a model on 28x28 greyscales images that belong to 1 of 10 categories: 
myCNN = MyCNN(
    width=28,
    height=28,
    channels=1,
    num_classes=10
)

# Load model with pre-trained weights:
# NOTE: Assumes weights have already been saved by running "example.py"
myCNN.loadWeights("mycnn-weights.h5")

# Prepare sample data:
test_samples = myCNN.prepareSamples(test_samples)

# Evaluate model to see if everything works:
myCNN.evalModel(test_samples, test_labels)
