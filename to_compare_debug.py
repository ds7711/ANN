# import modules
import numpy as np
import matplotlib.pylab as plt
import ann_lib as ann
import ann_lib_naive
import time

# load data
from mnist_loader import load_data_wrapper
training_data, validation_data, test_data = load_data_wrapper()

# test Network Object
my_network = ann.Network([len(training_data[0][0]), 100, 10], normalize=True, activation_func=ann.Activation_Func.poisson_func)
print my_network.size, my_network.num_neuron_layer
my_network.mini_batch_SGD(training_data, 60, 1, 10**-10, 10)