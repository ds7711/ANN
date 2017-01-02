# import modules
import numpy as np
import matplotlib.pylab as plt
import ann_lib as ann
import ann_lib_naive
import time

# load data
from mnist_loader import load_data_wrapper
training_data, validation_data, test_data = load_data_wrapper()

my_network = ann.Network([len(training_data[0][0]), 150, 10], normalize=True,
                         activation_func=ann.Activation_Func.poisson_func,
                         derivative_func=ann.Activation_Func.poisson_derivative)
my_network.mini_batch_SGD(training_data, test_data, 60, 1, 10**-8, 60)
# my_network.save_weights("poisson_network_weights.npz")


# test the effect of upper_bound in poission_derivative in learning
# ub_list = np.logspace(1, 3, 10, endpoint=True, base=10)
# ub_performance_list = []
# for ub in ub_list:
#     def poisson_derivative(weighted_sum, upper_bound=ub):
#         tmp = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
#         sigmoid_derivative = tmp * (1 - tmp)
#         mask = np.logical_and(weighted_sum > 0, weighted_sum < upper_bound)
#         return mask + np.logical_not(mask) * sigmoid_derivative
#
#     test_network = ann.Network([len(training_data[0][0]), 100, 10], normalize=True,
#                                activation_func=ann.Activation_Func.poisson_func,
#                                derivative_func=poisson_derivative)
#     ub_performance_list.append(test_network.mini_batch_SGD(training_data, test_data, 60, 0.1, 10**-10, 10))
#
# for idx in xrange(len(ub_list)):
#     accuracies = ub_performance_list[idx]
#     plt.figure()
#     plt.plot(np.arange(10)+1, accuracies[0], color="b", marker="o")
#     plt.plot(np.arange(10)+1, accuracies[1], color="r", marker="*")

# a good value of upper_bound of poission derivative function is between 10 to 30


# relu_network = ann.Network([len(training_data[0][0]), 100, 10], normalize=True,
#                            activation_func=ann.Activation_Func.sigmoid_func, derivative_func=ann.Activation_Func.sigmoid_derivative)
# relu_network.mini_batch_SGD(training_data, 60, 0.1, 1e-5, 10)

# ex_image, ex_digit = training_data[1]
# ann.digit_visualize(ex_image)
# print "The input number is: %d" % np.flatnonzero(ex_digit)
# plt.close()

# test Network Object
# my_network = ann.Network([len(training_data[0][0]), 100, 10], normalize=True,
#                          activation_func=ann.Activation_Func.poisson_func, derivative_func=ann.Activation_Func.poisson_derivative)
# print my_network.size, my_network.num_neuron_layer
# my_network.mini_batch_SGD(training_data, test_data, 60, 0.1, 10**-10, 10)
# print my_network.test_accuracy(test_data)

# test activations
# filename = "pre_network.npz"
# np.savez_compressed(filename, my_network._weight_matrix_list)
# my_network.save_weights(filename)
#
# # load the pre-trained network
# weights_list = np.load(filename)
# weights_list = weights_list["arr_0"]
# weights_list = weights_list[()]
# # weights_list = [weights_list[0], weights_list[1]]
# my_network.load_weights(weights_list)
#
# my_network.test_accuracy(test_data)
# my_network.training_accuracy(training_data)

# print my_network.mini_batch_SGD(training_data, 60, 0.1, 0, 30)
# my_network.mini_batch_SGD(training_data, 30, 1, 10**-4, 10)
# my_network.test_accuracy(test_data)

# relu_network = ann.Network([len(training_data[0][0]), 100, 10], normalize=True,
#                            activation_func=ann.Activation_Func.relu_func, derivative_func=ann.Activation_Func.relu_derivative)
# relu_network.mini_batch_SGD(training_data, 30, 0.01, 10**-4, 3)



# check weights
# idx = 0
# print np.sum(my_network._weight_matrix_list[idx] - naive_network.weights[idx])
#
# # test activations
# mini_batch = training_data[0:3]
# input_matrix = []
# y_matrix = []
# for input_vec, y_vec in mini_batch: # convert input data to desirable format
#     input_matrix.append(input_vec.squeeze())
#     y_matrix.append(y_vec.squeeze())
# input_matrix = np.array(input_matrix).T
# y_matrix = np.array(y_matrix).T
# activation_matrix_list, derivative_matrix_list = my_network._forward_pass(input_matrix)
#
# # test activation function
# weighted_sum = 9
# print ann.Activation_Func.sigmoid_func(weighted_sum)
# print ann.Activation_Func.sigmoid_derivative(weighted_sum)
# print ann.Activation_Func.relu_func(weighted_sum+1)
# print ann.Activation_Func.relu_derivative(weighted_sum+1)