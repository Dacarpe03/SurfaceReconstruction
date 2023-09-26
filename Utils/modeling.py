import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_linear_architecture(
	input_shape,
	output_size,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation
	use_batch_normalization=True,
	name="SurfaceReconstructor"
	):

	"""
	Defines de architecture of the neural network
	Input:
		input_shape (tuple): The shape a data point in the features dataset
		output_size (int): The length of a data point in the labels dataset
		hidden_layer_sizes (list): A list of integers
		regularizer (keras.regularizers): A regularizer for the hidden layers (e.g. L1, see keras documentation for more)
		hidden_activation (string): The name of the activation function of the hidden layers' neurons  (e.g 'relu', see keras documentation for more)
		output_activation (string): The name of the activation function of the output layers (e.g 'linear', see keras documentation for more)
		use_batch_normalization (bool): If True, then add batch normalization to the hidder layers
	"""

	# Create a sequential model
    model = keras.Sequential(name=name)

    # Create the first layer
    model.add(keras.layers.InputLayer(input_shape=input_shape,
                                      batch_size=None))
    
    # Create the hidden layers of the neural network
    for neurons in hidden_layer_sizes:
        model.add(keras.layers.Dense(neurons,
                                     kernel_regularizer=regularizer,
                                     kernel_initializer=keras.initializers.HeNormal(seed=None),
                                     use_bias=False
                                     ))
        if use_batch_normalization:
        	model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Activation(hidden_activation))
        
    model.add(keras.layers.Dense(output_size,
                                 activation=output_activation
                                ))
    return model