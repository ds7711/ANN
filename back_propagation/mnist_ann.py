# import modules
import numpy as np
import matplotlib.pylab as plt
import ann_lib_naive as ann
import time


# load data 
from mnist_loader import load_data_wrapper
training, validation, test = load_data_wrapper()
ex_image, ex_digit = training[1]
ann.digit_visualize(ex_image)
print "The input number is: %d" % np.flatnonzero(ex_digit)
plt.close()

# initialize the neural network
Sigmoid_Network = ann.Network([len(ex_image), 100, 10])
print Sigmoid_Network.num_layer
# print Sigmoid_Network.weights
# before = np.copy(Sigmoid_Network.weights[1])
st_time = time.time()
Sigmoid_Network.mini_batch_SGD(training, validation, 30, 3, 10)
print "The trianing takes: %f seconds." % (time.time() - st_time)

filename = "pre_network.npz"
# Sigmoid_Network.save_weights(filename)

# load the pre-trained network
weights_list = np.load(filename)
weights_list = weights_list["arr_0"]
weights_list = weights_list[()]
weights_list = [weights_list[0], weights_list[1]]
Sigmoid_Network.load_weights(weights_list)
new_x = [(Sigmoid_Network.predict(image)[1][:-1], label) for image, label in training]
# weights_dict = np.load(filename)


# include heterogeneity: inhibotory neurons whose weights are all negative? 
Post_Network = ann.Network([10, 10])
Post_Network.mini_batch_SGD(new_x, new_x, 300, 0.1, 30, test_accuray=False)

# combine the networks
Combined_Network = ann.Network([len(ex_image), 100, 10, 10])
Combined_Network.load_weights(weights_list + Post_Network.weights)
Combined_Network.accuracy(test)
Combined_Network.mini_batch_SGD(training, test, 300, 0.1, 30)

# Performance of the combined network
corr_count = 0
for input_vec, label in test:
    corr_count += Combined_Network.predict(input_vec)[0] == label
print "The combined network has accuracy: %f" % (corr_count * 1.0 / len(test))

corr_count = 0
for input_vec, label in test:
    corr_count += Sigmoid_Network.predict(input_vec)[0] == label
print "The combined network has accuracy: %f" % (corr_count * 1.0 / len(test))
