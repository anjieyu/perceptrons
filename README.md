# perceptrons
A Python project implementing simple perceptron classifiers by hand

## How to run
1. Download or clone the Git repository with ```git clone https://github.com/anjieyu/perceptrons.git```
2. Navigate to the project directory with the command ```cd perceptrons```
3. Run ```python3 main.py```

## What this project does
### Data
The first thing this project does is load the training data from the data directory and store in a dictionary that maps all data points from each training data file and associates them with the file name that they came from.

The dataset is a simple randomly generated dataset of 2 classes which are labeled 0 or 1 and consist of an x, y coordinate. Samples with an x value less than 5 are class 0. Samples with an x value greater than 5 are class 1. The samples are randomly generated from a Gaussian distribution.

### Perceptrons
After the data is imported, we create a perceptron for each training dataset. The perceptron class is located in the file ```perceptron.py``` and has a constructor and 3 methods. 
- def \_\_init__(self, inputs, classes):
    - The constructor takes in the number of features as inputs for each sample, and the number of classes for which to predict for.
    - To initialize a perceptron, this constructor also initializes weights based on the number of inputs from a range from 0 to 3.
    - A bias term is added with a constant feature input value of 1.5 and a randomized weight from 0 to 1.
    - Finally, a learning rate is set for the perceptron with a value of 0.85.
- def activate(self, sample):
    - The activation function takes the values of the sample as the input and multiplies each term with the corresponding weight for the feature. It also multiplies the bias term with the current bias weight and sums together the result of all features times their feature weights plus the bias times the bias weight.
    - The resulting value is the interpreted result of the sample by the perceptron given the perceptron's current weights.
- def step(self, activation):
    - The step function is a simple step activation function that assigns a class to a sample if the activation is greater than 0. If the activation is greater than 0, the class is 1. If the activation is less than 0, then the class is 0.
- def learn(self, sample, label):
    - This function performs the weight updates for a perceptron after it receives a sample. It uses the activate and step functions to determine a class for a given sample.
    - If the predicted label does not match the actual label, then a weight update step will be performed on the perceptron, by adding or subtracting the current weight values times the sample features to the weights again.
    - The same is applied to the bias weight as well.
    - The learning rate is multiplied by the change in weight calculated from the previous step to normalize the weight update rule.
    - If the predicted label matches the actual label, then no weight update will be performed.

Each perceptron for each training set of data learns per sample, so the weight update rule is applied for each misclassification on a 1 to 1 basis. The decision boundary is updated for each misclassification per sample in each training set.