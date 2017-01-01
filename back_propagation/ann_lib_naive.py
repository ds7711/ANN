import numpy as np
import matplotlib.pylab as plt
import random


def digit_visualize(data_vector):
    """
    visualize the digit image
    :param data_vector: 
    :return: 
    """
    pixels = int(np.sqrt(len(data_vector)))
    plt.figure()
    plt.pcolormesh(np.reshape(data_vector, (pixels, pixels)), cmap="gray")
    plt.gca().invert_yaxis()


class Network(object):
    def __init__(self, network_size):
        """
        initialize the network
        :param network_size: a list specify # of neurons in each layer, from input to output layer \n
        weights: a list of matrix for each layer, excluding input layer \n
            for weight matrix of layer (l+1), each row specifies connection weights from neurons in previous layer(l), \
            including weight for bias (the last column)
        """
        self.num_layer = len(network_size)
        self.weights = []
        for iii in xrange(self.num_layer - 1):
            weight_matrix = np.random.randn(network_size[iii+1]+1, network_size[iii]+1) # augment the weight matrix to include weight for bias
            self.weights.append(weight_matrix)

    def __activation_func(self, weighted_sum):
        """
        activation function: e.g., sigmoid function;
        :param weighted_sum: the weighted sum from previous layer
        :return: activation
        """
        return 1.0 / (1 + np.exp(-1.0 * weighted_sum))

    def __activation_derivative(self, weighted_sum):
        tmp = self.__activation_func(weighted_sum)
        return (1 - tmp) * tmp

    def __forward_pass(self, vectorized_digit, label):
        weighted_sums = []
        derivatives = [] # the derivative of activation with respect to input for each neuron
        input_activation = np.append(vectorized_digit, [1]) # append a 1 to the end of vectorized input
        activations = [input_activation]
        for iii in xrange(self.num_layer-1):
            tmp_weighted_sum = np.dot(self.weights[iii], input_activation)
            weighted_sums.append(tmp_weighted_sum)
            input_activation = np.array(self.__activation_func(tmp_weighted_sum))
            input_activation[-1] = 1
            activations.append(input_activation)
            # neuron_derivative = np.array(self.__activation_derivative(input_activation))
            neuron_derivative = np.array(self.__activation_derivative(tmp_weighted_sum))
            neuron_derivative[-1] = 0 # the fictive neuron has activation 0 and derivative 0
            derivatives.append(neuron_derivative)
        output_errors = (input_activation - np.append(label, [1])) * neuron_derivative #
        return activations, derivatives, output_errors

    def __back_propagation(self, vectorized_digit, label):
        activations, derivatives, output_errors = self.__forward_pass(vectorized_digit, label)
        errors = [0] * (self.num_layer - 1)
        errors[-1] = output_errors
        for iii in xrange(self.num_layer-3, -1, -1):
            post_error = errors[iii+1]
            pre_error = np.dot(post_error, self.weights[iii+1]) * derivatives[iii]
            errors[iii] = pre_error
        return errors, activations

    def __weight_changes(self, vectorized_digit, label):
        errors, activations = self.__back_propagation(vectorized_digit, label)
        weight_changes = []
        for iii in xrange(self.num_layer-1):
            tmp_change = np.dot(errors[iii][:, np.newaxis], activations[iii][np.newaxis, :])
            weight_changes.append(tmp_change)
        return weight_changes

    def __weight_update(self, training_data, learning_rate):
        weight_change_list = []
        for input_vec, label in training_data:
            weight_change_list.append(self.__weight_changes(input_vec, label))
        # print weight_change_list[1]
        for iii in xrange(self.num_layer-1):
            weight_change = []
            for jjj in xrange(len(weight_change_list)):
                weight_change.append(weight_change_list[jjj][iii])
            weight_change = np.mean(weight_change, axis=0)
            self.weights[iii] -= weight_change * learning_rate

    def mini_batch_SGD(self, training_data, test_data, batch_size, learning_rate=0.1, num_iteration=30, test_accuray=True):
        for idx in xrange(0, num_iteration+1):
            random.shuffle(training_data) # in place shuffle
            mini_batches = [training_data[k:k+batch_size] for k in xrange(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                self.__weight_update(mini_batch, learning_rate)
            if test_accuray and idx % 1 == 0:
                print "Iteration = %d, accuracy = %f\n" % (idx, self.accuracy(test_data))
                # print "The current accuracy is: %f" % self.accuracy(test_data)

    def accuracy(self, test_data):
        correct_count = 0
        for item in test_data:
            input_vec, label = item
            prediction, _ = self.predict(input_vec)
            correct_count += prediction == label
        return correct_count * 1.0 / len(test_data)

    def predict(self, vectorized_digit):
        input_activation = np.append(vectorized_digit, [1]) # append a 1 to the end of vectorized input
        for iii in xrange(self.num_layer-1):
            tmp_weighted_sum = np.dot(self.weights[iii], input_activation)
            input_activation = self.__activation_func(tmp_weighted_sum)
        return np.argmax(input_activation[:-1]), input_activation

    def load_weights(self, weights_list):
        for idx, weights in enumerate(weights_list):
            self.weights[idx] = weights

    def save_weights(self, filename):
        weights_dict = {}
        for idx, weights in enumerate(self.weights):
            weights_dict[idx] = weights
        np.savez_compressed(filename, weights_dict)



