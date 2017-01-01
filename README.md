#Repository to learn neural network.

"ann_lib" implemented back-progapation algorithm based on numpy (vectorized computation). The library allows one to specify number of neuronal layers and type of activation function used during the training. It includes one original poisson-based activation function that helps to fight against over-fitting, which is inspired by the properties of real neuron and dropout idea.

Scripts are tested using MNIST data set and achieved a 95% accuracy on the test set, without fining tuning the hyperparameters based on cross-validation.

I also showed how back-propagation works in terms of dynamic programming. However, this proof may not be self-contained if one is not familiar with multi-variate calculus or neural network per se. If interested, I would suggest to read the online deep learning book from Michael Nielsen first. 

Happy training! 
