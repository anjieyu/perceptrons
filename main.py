import os
import numpy as np
from matplotlib import pyplot as plt
from perceptron import Perceptron

# prepare structures to hold training and test data
training_data = {}
training_files = []
testing_data = []

# prepare the data path
data_path = os.path.join(os.getcwd(), 'data')

# read data from data path
if os.path.exists(data_path) and os.path.isdir(data_path):
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)

            # training files
            if file.endswith('.train'):
                training_set = [line.strip() for line in open(file_path)]
                for i, entry in enumerate(training_set):
                    sample = entry.split(' ')
                    x, y, label = sample
                    sample = (float(x), float(y), float(label))
                    training_set[i] = sample
                training_data[file] = training_set
                training_files.append(file)

            # test file
            elif file.endswith('.test'):
                testing_data = [line.strip() for line in open(file_path)]
                for i, entry in enumerate(testing_data):
                    sample = entry.split(' ')
                    x, y, label = sample
                    sample = (float(x), float(y), float(label))
                    testing_data[i] = sample
            
            else:
                print('file is not found')

training_files.sort()
temp = training_files[1]
del training_files[1]
training_files.append(temp)
print(training_files)

def plot_points(training_data, file):
    x_data, y_data, labels = zip(*training_data[file])
    graph_name = f'graphs/{file}_points.png'
    label0_samples = [i for i in range(len(labels)) if labels[i] == 0]
    label1_samples = [i for i in range(len(labels)) if labels[i] == 1]
    label0_x = [x_data[i] for i in label0_samples]
    label0_y = [y_data[i] for i in label0_samples]
    label1_x = [x_data[i] for i in label1_samples]
    label1_y = [y_data[i] for i in label1_samples]

    fig, ax = plt.subplots()
    ax.set_xlim([min(x_data) - 1, max(x_data) + 1])
    ax.set_ylim([min(y_data) - 1, max(y_data) + 1])
    plt.scatter(label0_x, label0_y, c='red')
    plt.scatter(label1_x, label1_y, c='blue')
    plt.savefig(graph_name)
    plt.close()

def plot_perceptron(training_data, file, perceptron, iteration):
    if 'test' in file:
        x_data, y_data, labels = zip(*training_data)
    else:
        x_data, y_data, labels = zip(*training_data[file])
    graph_name = f'graphs/{file}_perceptron_{iteration}.png'
    label0_samples = [i for i in range(len(labels)) if labels[i] == 0]
    label1_samples = [i for i in range(len(labels)) if labels[i] == 1]
    label0_x = [x_data[i] for i in label0_samples]
    label0_y = [y_data[i] for i in label0_samples]
    label1_x = [x_data[i] for i in label1_samples]
    label1_y = [y_data[i] for i in label1_samples]

    fig, ax = plt.subplots()
    ax.set_xlim([min(x_data) - 1, max(x_data) + 1])
    ax.set_ylim([min(y_data) - 1, max(y_data) + 1])
    plt.scatter(label0_x, label0_y, c='red')
    plt.scatter(label1_x, label1_y, c='blue')
    x_intercept = -(perceptron.bias_weight[0][0] * perceptron.bias) / perceptron.weights[0]
    y_intercept = -(perceptron.bias_weight[0][0] * perceptron.bias) / perceptron.weights[1]
    ax.axline((x_intercept, 0), (0, y_intercept))
    plt.savefig(graph_name)
    plt.close()

myPerceptron = Perceptron(2, 2)
# plot_perceptron(training_data, 'set1.train', myPerceptron)
mySample = training_data['set1.train'][0]
x, y, label = mySample
print(x, y)
print(myPerceptron.weights)
prediction = myPerceptron.learn((x, y), label)
# plot_perceptron(training_data, 'set1.train', myPerceptron)
print(prediction, label)
print(myPerceptron.weights)

for i, file in enumerate(training_files):
    num_correct = 0
    myPerceptron = Perceptron(2, 2)
    print('Training perceptron for file', file)
    for epoch in range(5):
        print('Epoch:', epoch)
        num_correct = 0
        for sample in training_data[file]:
            x, y, label = sample
            prediction = myPerceptron.learn((x, y), label)
            if prediction == label:
                num_correct += 1
            # print(prediction, label)
        print('Training accuracy: ', num_correct / len(training_data[file]))
        myPerceptron.learning_rate = np.square(myPerceptron.learning_rate)
        if num_correct / len(training_data[file]) == 1:
            print('100%% accuracy reached, early stopping')
            break
    plot_perceptron(training_data, file, myPerceptron, i+1)

    num_correct = 0
    for sample in testing_data:
        x, y, label = sample
        activation = myPerceptron.activate((x, y))
        prediction = myPerceptron.step(activation)
        if prediction == label:
            num_correct += 1
    print('Testing accuracy: ', num_correct / len(testing_data))
    plot_perceptron(testing_data, 'set.test', myPerceptron, i+1)

