# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cPickle
from itertools import chain
from load_images import read_tensor_convert
from PIL import Image
from load_images import read_images_direct
from load_images import read_image_files
from load_images import read_image_files_v2
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5))
       ])


#training_folders = ['/Users/norahborus/Documents/DATA/training_data/', "CROHME_training_2011/", "TrainINKML_2013/",
#                    "trainData_2012_part1/", "trainData_2012_part2/"]



def load_data():
    path = '/Users/norahborus/Documents/DATA/COMBINED/'
    folders = ['/Users/norahborus/Documents/DATA/COMBINED/', 'TRAIN_Characters/', 'TEST_Characters/']
    temp_train_x, temp_train_y, temp_test_x, temp_test_y, y_map = read_image_files_v2(folders)
    print "YMap: {}".format(y_map)
    with open(path + "y_map", "w") as fp:
        cPickle.dump(y_map, fp)

    train_data = []
    test_data = []
    classes = list(sorted(y_map, key=y_map.get))
    print "Classes: {}".format(classes)

    for image_file, label in zip(temp_train_x, temp_train_y):
        png = Image.open(image_file)
        image = png.convert('RGB')
        img_tensor = transform(image)
       # print img_tensor.size()
        train_data.append((img_tensor, y_map[label]))

    for image_file, label in zip(temp_test_x, temp_test_y):
        png = Image.open(image_file)
        image = png.convert('RGB')
        img_tensor = transform(image)
        test_data.append((img_tensor, y_map[label]))

    print "DONE"
    return train_data, test_data, classes, y_map





#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

import torch.optim as optim
import tensorflow as tf

def normalize(Y):
    min_val  = np.min(Y)
  #  print "Y before: {}".format(Y)
  #  print min_val
    Y = Y + abs(min_val)
  #  print "Y after: {}".format(Y)
    return 1.0*Y / np.sum(Y)


import cPickle

def train(train_data, classes):

    net = Net(classes)

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize

    sess = tf.Session()
    for epoch in range(30):  # loop over the dataset multiple times
        print "Epoch: {}".format(epoch)
        running_loss = 0.0
        print len(train_data)
        for i, data in enumerate(train_data):
            input,label = data
            one_d = torch.Tensor(1, 3, 32,32)
            labels = torch.Tensor(1, ).long()
            one_d[0] = input
            labels[0] = label
            # print labels.size()
            '''
            for i, tensor in enumerate(input):
                # print "Initial: ",type(tensor)
                rgb = tensor.cpu().numpy()
                # print rgb.shape
                rgb = np.swapaxes(rgb,0,2)
                rgb = np.swapaxes(rgb,0,1)
                # print rgb.shape
                a = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
                #  print a.shape
                tf_a = torch.from_numpy(a).float()
                one_d[i] = tf_a
            '''
            inputs = one_d
            # print inputs.size()
            # print "SECOND: ", inputs.size(), type(inputs), type(inputs[0])
            # inputs = torch.from_numpy(one_d).float()
            # print inputs[0].size()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # print inputs.size()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            #  print outputs
            #  print labels
           # print outputs, labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

    with open('cnn', 'w') as fp:
        cPickle.dump(net, fp)

    return net




def test_single_example(image, net, classes):
    one_d = torch.Tensor(1, 3, 32, 32)
    one_d[0] = image
    images = one_d
    images = Variable(images)
    outputs = net(images)
    np_array = outputs.data.cpu().numpy()
    np_array = normalize(np_array)
    #   print "Done -->{}".format(i)
    print "OUTPUTS: {}".format(np_array)
    return outputs


def test(test_data, net, classes):
    ########################################################################
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.


    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    sess = tf.Session()
    for image, label in test_data:
        one_d = torch.Tensor(1, 3, 32, 32)
        labels = torch.Tensor(1, ).long()
        one_d[0] = image
        labels[0] = label
        # print labels.size()
        '''
        for i, tensor in enumerate(input):
            # print "Initial: ",type(tensor)
            rgb = tensor.cpu().numpy()
            # print rgb.shape
            rgb = np.swapaxes(rgb,0,2)
            rgb = np.swapaxes(rgb,0,1)
            # print rgb.shape
            a = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            #  print a.shape
            tf_a = torch.from_numpy(a).float()
            one_d[i] = tf_a
        '''
        images = one_d
        # print inputs.size()
        # print "SECOND: ", inputs.size(), type(inputs), type(inputs[0])
        # inputs = torch.from_numpy(one_d).float()
        # print inputs[0].size()
        # wrap them in Variable
        images = Variable(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the {} test images: {}%'.format(len(test_data), 100 * correct / total))

    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    print len(test_data)

    for image, label in test_data:
        one_d = torch.Tensor(1, 3, 32, 32)
        labels = torch.Tensor(1, ).long()
        one_d[0] = image
        labels[0] = label
        # print labels.size()
        '''
        for i, tensor in enumerate(input):
            # print "Initial: ",type(tensor)
            rgb = tensor.cpu().numpy()
            # print rgb.shape
            rgb = np.swapaxes(rgb,0,2)
            rgb = np.swapaxes(rgb,0,1)
            # print rgb.shape
            a = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            #  print a.shape
            tf_a = torch.from_numpy(a).float()
            one_d[i] = tf_a
        '''
        images = one_d
        # print inputs.size()
        # print "SECOND: ", inputs.size(), type(inputs), type(inputs[0])
        # inputs = torch.from_numpy(one_d).float()
        # print inputs[0].size()
        # wrap them in Variable
        images = Variable(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        label = labels[0]
        class_correct[label] += c[0]
        class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] == 0:
            continue
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def main():
    train_data, test_data, classes, y_map = load_data()
    net = train(train_data, classes)
   # return net, classes
    test(test_data, net, classes)


if __name__ == '__main__':
    main()

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor on to the GPU, you transfer the neural
# net onto the GPU.
# This will recursively go over all modules and convert their parameters and
# buffers to CUDA tensors:
#
# .. code:: python
#
#     net.cuda()
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ::
#
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train an face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train an face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
