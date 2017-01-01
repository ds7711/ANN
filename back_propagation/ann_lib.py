import numpy as np
import matplotlib.pylab as plt
import random


class Activation_Func(object):
    @staticmethod
    def sigmoid_func(weighted_sum):
        return 1.0 / (1 + np.exp(-1.0 * weighted_sum))

    @staticmethod
    def sigmoid_derivative(weighted_sum):
        tmp = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
        return tmp * (1 - tmp)

    @staticmethod
    def relu_func(weighted_sum, leaky_factor=0.05):
        return np.abs(np.clip(((weighted_sum > 0) + (weighted_sum <= 0) * leaky_factor) * weighted_sum, -2, 2)) # for matrix relu

    @staticmethod
    def relu_derivative(weighted_sum, leaky_factor=0.05):
        return np.array(weighted_sum > 0, dtype=float) + (weighted_sum <= 0) * leaky_factor # for matrix relu

    @staticmethod
    def huber_func(weighted_sum):
        sigmoid_activation = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
        activation = np.clip(((weighted_sum>0) * weighted_sum + sigmoid_activation), 1e-10, 1-1e-10)
        return activation

    @staticmethod
    def huber_derivative(weighted_sum):
        tmp = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
        sigmoid_derivative = tmp * (1 - tmp)
        return np.array(weighted_sum > 0, dtype=float) + (weighted_sum <= 0) * sigmoid_derivative

    @staticmethod
    def poisson_func(weighted_sum):
        avg_fr = np.clip(np.ceil(weighted_sum), 1, 10)
        fr = np.random.poisson(avg_fr)
        sigmoid_activation = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
        return sigmoid_activation * (weighted_sum < 0) + fr * (weighted_sum > 0)
        # return fr * (weighted_sum > 0)

    @staticmethod
    def poisson_derivative(weighted_sum, upper_bound=10):
        tmp = 1.0 / (1 + np.exp(-1.0 * weighted_sum))
        sigmoid_derivative = tmp * (1 - tmp)
        mask = np.logical_and(weighted_sum > 0, weighted_sum < upper_bound)
        return mask + np.logical_not(mask) * sigmoid_derivative

class Cost_Func(object):
    @staticmethod
    def cross_entropy(prediction_vec, y_vec):
        error_vec = y_vec * np.log(prediction_vec) + (1 - y_vec) * np.log(1 - prediction_vec)
        return np.dot(error_vec, error_vec)


class Network(object):
    def __init__(self, network_size, normalize=True,
                 activation_func=Activation_Func.sigmoid_func,
                 derivative_func=Activation_Func.sigmoid_derivative):
        """
        initialize the network with random weights
        :param network_size: a list specify the number of neurons in each layer. doesn't include the extra neuron for bias.\n
        :param normalize: whether normalize the weight to prevent "saturation".
        """
        self.size = network_size
        self.num_neuron_layer = len(network_size) - 1 # excluding input layer
        self._weight_matrix_list = self._weight_initialize(normalize)
        self.activation_func = activation_func
        self.derivative_func = derivative_func

    def _weight_initialize(self, normalize):
        weight_matrix_list = []
        if normalize:
            for iii in xrange(self.num_neuron_layer):
                weight_matrix = np.random.randn(self.size[iii+1]+1, self.size[iii]+1) / np.sqrt(self.size[iii])
                weight_matrix_list.append(weight_matrix)
        else:
            for iii in xrange(self.num_neuron_layer):
                # add 1 dimension to include bias
                weight_matrix = np.random.randn(self.size[iii+1]+1, self.size[iii]+1)
                weight_matrix_list.append(weight_matrix)
        return weight_matrix_list

    def _forward_pass(self, input_matrix):
        """
        calculate the activations for all neurons over all training examples.\n
        :param input_matrix: 2D numpy array, p * n, p is the dimension of input data, n: number of training examples. \n
        :return: list of activations at each layer.\n
            e.g., the first matrix in the list is the activations of neurons in the first layer from n training examples. \n
        """
        # add a constant 1 to the input matrix to include the bias in the weight matrix
        input_matrix = np.vstack((input_matrix, np.ones(input_matrix.shape[1])))
        weighted_sums_list = []
        derivative_matrix_list = []
        activation_matrix_list = [input_matrix]
        for iii in xrange(self.num_neuron_layer):
            # linear weighted sums: z
            weighted_sums = np.dot(self._weight_matrix_list[iii], input_matrix)
            weighted_sums_list.append(weighted_sums)
            # non-linear activations
            activation_matrix = np.array(self.activation_func(weighted_sums))
            activation_matrix[-1] = 1 # change the last row to 1, it's for bias
            input_matrix = activation_matrix
            activation_matrix_list.append(input_matrix)
            # derivatives
            derivative_matrix = np.array(self.derivative_func(weighted_sums))
            derivative_matrix[-1] = 0 # bias terms shouldn't contribute to the back-propagated errors
            derivative_matrix_list.append(derivative_matrix)
        # un-comment if use cross-entropy cost function
        # for the output layer, prevent it from becoming zero
        # input_matrix = np.clip(input_matrix, 1e-10, 1 - 1e-10) # add a constant to input matrix to prevent 0
        input_matrix[:-1, :] = 1.0 / (1 + np.exp(-1.0 * weighted_sums[:-1, :]))
        tmp = np.exp(input_matrix[:-1, :])
        input_matrix[:-1, :] = tmp / np.tile(tmp.sum(axis=0), (tmp.shape[0], 1)) # softmax output
        input_matrix[-1] = 1
        # activation_matrix_list[-1] = input_matrix
        return activation_matrix_list, derivative_matrix_list

    def _output_errors(self, output_activation_matrix, y_matrix, output_derivative_matrix):
        # cross-entropy error
        # output_error_matrix = -1.0 * (y_matrix / output_activation_matrix[:-1, :] -
        #                        (1 - y_matrix) / (1 - output_activation_matrix[:-1, :])) * output_derivative_matrix[:-1, :]
        # quadratic cost function
        output_error_matrix = (output_activation_matrix[:-1, :] - y_matrix) * output_derivative_matrix[:-1, :]
        output_error_matrix = np.vstack((output_error_matrix, np.zeros(y_matrix.shape[1])))
        return output_error_matrix

    def _back_propagation(self, output_error_matrix, derivative_matrix_list):
        error_matrix_list = [0] * self.num_neuron_layer
        error_matrix_list[self.num_neuron_layer-1] = output_error_matrix # the last item in the list output error matrix
        for iii in xrange(self.num_neuron_layer-2, -1, -1):
            weight_matrix = self._weight_matrix_list[iii+1]
            error_matrix = np.dot(weight_matrix.T, output_error_matrix) * derivative_matrix_list[iii]
            output_error_matrix = error_matrix
            error_matrix_list[iii] = error_matrix
        return error_matrix_list

    def _weight_update(self, error_matrix_list, activation_matrix_list, learning_rate, regularization_strength):
        for iii in xrange(len(error_matrix_list)):
            error_matrix = error_matrix_list[iii]
            activation_matrix = activation_matrix_list[iii]
            # delta_weight_matrix = np.dot(error_matrix.T[:, :, np.newaxis], activation_matrix.T[:, np.newaxis, :])
            sum_delta_weights = np.zeros((error_matrix.shape[0], activation_matrix.shape[0]))
            for jjj in xrange(error_matrix.shape[1]):
                sum_delta_weights += np.dot(error_matrix[:, jjj][:, np.newaxis], activation_matrix[:, jjj][np.newaxis, :])
            avg_delta_weights = sum_delta_weights / error_matrix.shape[1]
            self._weight_matrix_list[iii] = self._weight_matrix_list[iii] - learning_rate * avg_delta_weights - \
                                             learning_rate * regularization_strength * self._weight_matrix_list[iii]

    def mini_batch_SGD(self, training_data, test_data, batch_size, learning_rate, regularization_strength, num_iteration):
        training_accuracy_list = []
        test_accuracy_list = []
        for idx in xrange(0, num_iteration):
            random.shuffle(training_data) # in place shuffle
            mini_batches = [training_data[k:k+batch_size] for k in xrange(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                input_matrix = []
                y_matrix = []
                for input_vec, y_vec in mini_batch: # convert input data to desirable format
                    input_matrix.append(input_vec.squeeze())
                    y_matrix.append(y_vec.squeeze())
                input_matrix = np.array(input_matrix).T
                y_matrix = np.array(y_matrix).T
                activation_matrix_list, derivative_matrix_list = self._forward_pass(input_matrix)
                output_activation_matrix = activation_matrix_list[-1]
                output_derivative_matrix = derivative_matrix_list[-1]
                output_error_matrix = self._output_errors(output_activation_matrix, y_matrix, output_derivative_matrix)
                error_matrix_list = self._back_propagation(output_error_matrix, derivative_matrix_list)
                self._weight_update(error_matrix_list, activation_matrix_list, learning_rate, regularization_strength)
            train_accuracy = self.training_accuracy(training_data)
            test_accuracy = self.test_accuracy(test_data)
            print "Iteration = %d, training accuracy = %f, test accuracy = %f\n" % (idx + 1, train_accuracy, test_accuracy)
            training_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
        return np.array([training_accuracy_list, test_accuracy_list])


    def predict(self, vectorized_digit):
        input_activation = np.append(vectorized_digit, [1]) # append a 1 to the end of vectorized input
        for iii in xrange(self.num_neuron_layer):
            tmp_weighted_sum = np.dot(self._weight_matrix_list[iii], input_activation)
            input_activation = self.activation_func(tmp_weighted_sum)
        return np.argmax(input_activation[:-1]), input_activation

    def training_accuracy(self, test_data):
        correct_count = 0
        for item in test_data:
            input_vec, y_vec = item
            prediction, _ = self.predict(input_vec)
            correct_count += prediction == np.argmax(y_vec)
        return correct_count * 1.0 / len(test_data)

    def test_accuracy(self, test_data):
        correct_count = 0
        for item in test_data:
            input_vec, label = item
            prediction, _ = self.predict(input_vec)
            correct_count += prediction == label
        return correct_count * 1.0 / len(test_data)

    def load_weights(self, weights_list):
        for idx, weights in enumerate(weights_list):
            self._weight_matrix_list[idx] = weights

    def save_weights(self, filename):
        np.savez_compressed(filename, self._weight_matrix_list)



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


