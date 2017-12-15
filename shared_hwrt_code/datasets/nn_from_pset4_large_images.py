import numpy as np
import matplotlib.pyplot as plt

params = {}
num_classes = 0

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x, num_classes):
    # print "X shape: ", x.shape
     m,n = x.shape
     e_x = np.exp(x - np.max(x, axis=0))
   #  print e_x.shape
     class_sums = np.reshape(np.sum(e_x, axis=0), (1,n)) #1x1000
     class_sums = np.repeat(class_sums,num_classes, axis=0)
     s = e_x  / class_sums
     return s
   #  print "S shape: ",s.shap
    ### YOUR CODE HERE

    ### END YOUR CODE



def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = 1.0 / (1.0+np.exp(-x))
    ### END YOUR CODE
    return s




def forward_prop(data, labels, params, num_classes):
    """
    return hidder layer, output(softmax) layer and loss
    """

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    X = data
    Y = labels
    M, N = X.shape

    z1 = np.dot(W1, X.T) + b1
    h = sigmoid(z1)
    z2 = np.dot(W2,h) + b2
    Y_hat = softmax(z2, num_classes)
    cost = (-1.0/M) * np.sum(np.log(Y_hat.T)*Y)

    ### YOUR CODE HERE

    ### END YOUR CODE

    return h, Y_hat, cost


def backward_prop(data, labels, params, num_classes):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    X = data
    Y = labels
    M, N = X.shape

    z1 = np.dot(W1, X.T) + b1
    h = sigmoid(z1)
    z2 = np.dot(W2,h) + b2
    Y_hat = softmax(z2, num_classes)
    ### YOUR CODE HERE

    intermediate2 = Y_hat.T - Y
    gradW2 = ( (np.dot(h, intermediate2)).T ) * 1.0/M

    gradb2 = np.reshape(np.sum(intermediate2, axis=0).T, (num_classes,1)) * 1.0/M #10x1

    intermediate1 = np.dot(W2.T, intermediate2.T)*h*(1-h)  #300x1000

    gradW1 = np.dot(intermediate1, X)  * 1.0/M  #300x784

    gradb1 =np.reshape(np.sum(intermediate1, axis=1), (400,1)) * 1.0/M  #300x1
    ### END YOUR CODE


    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

import cPickle
import matplotlib.pyplot as plt


def nn_train(trainData, trainLabels, num_classes, devData=None, devLabels=None):
    (m, n) = trainData.shape
    num_hidden = 400
    learning_rate = 5
    params = {}
    batch_size = 100
    num_iterations = len(trainLabels) / batch_size

    W1 = np.random.normal(0,1, size=(num_hidden, n))
    W2 = np.random.normal(0,1, size=(num_classes, num_hidden))
    b1 = np.zeros((num_hidden, 1))
    b2 = np.zeros((num_classes, 1))
    params['W1'] = W1
    params['W2'] = W2
    params['b1'] = b1
    params['b2'] = b2
    train_loss_history = []
    dev_loss_history  = []
    train_accuracy_history = []
    dev_accuracy_history = []
    regularization_constant = 0.0001
    for epoch in range(5):
        for batch_index in range(num_iterations):
            batch_train_data = trainData[batch_index*batch_size:batch_index*batch_size+batch_size]
          #  print batch_train_data.shape
            batch_train_labels = trainLabels[batch_index*batch_size:batch_index*batch_size+batch_size]
         #   print batch_train_labels.shape
            grad = backward_prop(batch_train_data, batch_train_labels, params, num_classes)
          #  print "W1: {}".format(params['W1'])
         #   print "W2: {}".format(params['W2'])
            params['W1'] = params['W1'] - learning_rate * (grad['W1'] + 2 * regularization_constant * params['W1'])
            params['W2'] = params['W2'] - learning_rate * (grad['W2'] + 2 * regularization_constant * params['W2'])
            params['b1'] = params['b1'] - learning_rate * grad['b1']
            params['b2'] = params['b2'] - learning_rate * grad['b2']

        h, Y_hat_train, train_loss = forward_prop(trainData, trainLabels, params, num_classes)
       # h, Y_hat_dev, dev_loss = forward_prop(devData, devLabels, params, num_classes)
        train_accuracy = compute_accuracy(Y_hat_train.T, trainLabels)
       # dev_accuracy = compute_accuracy(Y_hat_dev.T, devLabels)
        train_accuracy_history.append(train_accuracy)
       # dev_accuracy_history.append(dev_accuracy)
        train_loss_history.append(train_loss)
      #  dev_loss_history.append(dev_loss)
        print "Epoch: {}, Train loss: {}, Train Accuracy: {} ******".format(epoch+1,
                                                                                    train_loss, train_accuracy,
                                                                                 )

    '''
    for param in params:
        with open("regularized_{}".format(param), "w") as fp:
            cPickle.dump(params[param], fp)
    '''

    with open("nn_accuracy", "w") as fp:
        cPickle.dump(train_accuracy_history, fp)
    '''
    train_loss, = plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss', color='red')
   # dev_loss, = plt.plot(range(len(dev_loss_history)), dev_loss_history, label='Dev Loss', color='green')
    plt.legend([train_loss], ['Train Loss'])
    plt.savefig('loss_with_regularization.png')

    plt.clf()

    train_accuracy, = plt.plot(range(len(train_accuracy_history)), train_accuracy_history, label='Train Accuracy', color='red')
   # dev_accuracy, = plt.plot(range(len(dev_accuracy_history)),dev_accuracy_history , label='Dev Accuracy', color='green')
    plt.legend([train_accuracy], ['Train Accuracy'])
    plt.savefig('accuracy_with_regularization.png')

    '''
    ### YOUR CODE HERE


    ### END YOUR CODE

    return params


def nn_test(data, labels, params, num_classes):
    h, output, cost = forward_prop(data, labels, params, num_classes)
    accuracy = compute_accuracy(output.T, labels)
    return accuracy


def predict(data, params, num_classes):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    X = data
    M, N = X.shape

    z1 = np.dot(W1, X.T) + b1
    h = sigmoid(z1)
    z2 = np.dot(W2, h) + b2
    Y_hat = softmax(z2, num_classes)

    return Y_hat.T

def compute_accuracy(output, labels):
    print output.shape, labels.shape
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy * 100


def one_hot_labels(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels





def main():
    from PIL import Image
    from parallelism_read import read_images_direct
    from parallelism_read import read_image_files
    from parallelism_read import read_image_files_v2
    from parallelism_read import read_images_flattened
    from parallelism_read import read_images_flattened_v2
    global params
    global num_classes
    np.random.seed(100)
   # folders = ['/Users/norahborus/Documents/DATA/training_data/', '12_Characters/', 'TEST_Characters/']
    folders = ['/Users/norahborus/Documents/DATA/training_data/200x200/', 'CROHME_Characters/', 'CROHME_Characters/']
    trainData ,trainLabels, testData, testLabels, y_map = read_images_flattened(folders, segment=False)
    trainData = np.array(trainData)
    testData = np.array(testData)
    num_classes = len(y_map)

    with open("cseg_y_map", "w") as fp:
        cPickle.dump(y_map, fp)

    with open("cseg_num_classes", "w") as fp:
        cPickle.dump(num_classes, fp)

 #   return y_map, num_classes
    m, n = trainData.shape
    updatedTrainLabels= []
    updatedTestLabels = []
    # print TRAINING_Y
    for y in trainLabels:
        updatedTrainLabels.append(y_map[y])

    for y in testLabels:
        updatedTestLabels.append(y_map[y])

    testLabels = updatedTestLabels
    trainLabels = updatedTrainLabels

    testLabels = np.array(testLabels)
    trainLabels = np.array(trainLabels)

    trainLabels = one_hot_labels(trainLabels, num_classes)
    print trainData.shape
    print trainLabels.shape
    print "Ymap: ", len(y_map)
    print "Here: traindata shape: {}".format(trainData.shape)
    p = np.random.permutation(m)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    '''
    devData = trainData[0:m/10, :]
    devLabels = trainLabels[0:m/10, :]
    trainData = trainData[m/10:, :]
    trainLabels = trainLabels[m/10:, :]
    '''
    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
   # devData = (devData - mean) / std

    testLabels = one_hot_labels(testLabels, num_classes)
    print testLabels.shape
  #  print devLabels.shape
    print trainLabels.shape
    testData = (testData - mean) / std
    params = nn_train(trainData, trainLabels, num_classes)
    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params, num_classes)
        print 'Test accuracy: %f' % accuracy

    return params, num_classes, y_map


if __name__ == '__main__':
    main()