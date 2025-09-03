import numpy as np

class Perceptron:
    def __init__(self, inputs, classes):
        self.inputs = inputs
        self.classes = classes
        self.weights = np.random.uniform(low=0, high=3, size=(self.inputs,))
        self.bias = 1.5
        self.bias_weight = np.random.rand(1, 1)
        self.learning_rate = 0.85

    def activate(self, sample):
        activation = 0
        for i, feature in enumerate(sample):
            activation += feature * self.weights[i]
        activation += np.multiply(self.bias, self.bias_weight)
        return activation
    
    def step(self, activation):
        return int(activation >= 0)
    
    def learn(self, sample, label):
        activation = self.activate(sample)
        prediction = self.step(activation)
        if prediction < label:
            # Predicted 0 but actual is 1
            self.weights = np.add(self.weights, np.multiply(sample, self.learning_rate))
            self.bias_weight += self.bias
            pass
        elif prediction > label:
            # Predicted 1 but actual is 0
            self.weights = np.subtract(self.weights, np.multiply(sample, self.learning_rate))
            self.bias_weight -= self.bias
            pass
        else:
            # Prediction is correct
            pass
        self.bias += self.learning_rate * (label - prediction)
        return prediction