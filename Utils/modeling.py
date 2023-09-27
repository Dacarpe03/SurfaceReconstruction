import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from constants import MODELS_FOLDER_PATH


def read_data_for_training(
	features_file_path,
	labels_file_path,
	train_perc=0.8,
	val_perc=0.1,
	test_perc=0.1):
	"""
	Function to read and split our data

	Input:
		features_file_path (string): The .npy file with the features dataset
		labels_file_path (string): The .npy file with the labels dataset
		train_perc (float): Default 0.8. The percentage of the dataset used for training the dataset
		val_perc (float): Default 0.1. The percentage of the dataset used for validation the dataset
		test_perc (float): Default 0.1. The percentage of the dataset used for test the dataset
		
	Returns:
		train_features (np.array): An np.array containing np.array with the train features
		train_labels (np.array): An np.array containing np.array with the train features
		val_features (np.array): An np.array containing np.array with the train features
		val_labels (np.array): An np.array containing np.array with the train features
		test_features (np.array): An np.array containing np.array with the train features
		test_labels (np.array): An np.array containing np.array with the train features
	"""

	# Read the files with the dataset
	features = np.load(features_file_path, allow_pickle=True)
	labels = np.load(labels_file_path, allow_pickle=True)
	
	# Compute subset sizes
	total_rows = len(features)
	train_size = int(train_perc * total_rows)
	val_size = int(val_perc * total_rows)

	# Split the data based on sizes
	train_features = features[:train_size]
	train_labels = labels[:train_size]

	val_features = features[train_size:train_size + val_size]
	val_labels = labels[train_size:train_size + val_size]

	test_features = features[train_size + val_size:]
	test_labels = labels[train_size + val_size:]

	return train_features, train_labels, val_features, val_labels, test_features, test_labels


def create_linear_architecture(
	input_shape,
	output_size,
	hidden_layer_sizes,
	regularizer,
	hidden_activation,
	output_activation,
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
		name (string): The name of the model

	Returns:
		model (keras.Sequential): A keras neural network model with the architecture specified
	"""

	# Create a sequential model
	model = keras.Sequential(name=name)

	# Create input layer
	model.add(keras.layers.InputLayer(input_shape=input_shape,
																		batch_size=None))

	# Create the hidden layers of the neural network
  for neurons in hidden_layer_sizes:

  	model.add(keras.layers.Dense(neurons,
  															 kernel_regularizer=regularizer,
  															 kernel_initializer=keras.initializers.HeNormal(seed=None),
  															 use_bias=False))
  	if use_batch_normalization:
  		model.add(keras.layers.BatchNormalization())

  	model.add(keras.layers.Activation(hidden_activation))

  model.add(keras.layers.Dense(output_size,
  														 activation=output_activation))

 	return model



def compile_linear_model(
	model,
	loss_function,
	optimizer,
	metric
	):
  """
	Tells the model how to train
	
	Input:
		model (keras.Sequential): The sequential model to compile
		loss_function (keras.losses): The loss function used to update the weights of the neurons (eg. MeanSquaredError())
		optimizer (keras.optimizers): The optimizer used to update the weights of the neurons (eg. Adam)
		metric (keras.metrics): The metrics to monitor during the traning (eg. MeanSquaredError())

	Returns:
		None
  """
  model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=[metric])


def train_linear_model(
	model,
	train_features,
	train_labels,
	batch_size,
	epochs,
	val_features,
	val_labels,
	callbacks
	):
	
	"""
	Fits the model to the train instances of the data.

	Input:
		model (keras.Sequential): The sequential model to train
		train_features (np.array): An np.array containing np.array with the train features
		train_labels (np.array): An np.array containing np.array with the train features
		batch_size(int): The batch size of training samples used before each weight update
		epochs (int): The number of times the training goes through the training data
		val_features (np.array): An np.array containing np.array with the train features
		val_labels (np.array): An np.array containing np.array with the train features 
		callbacks (list): A list of keras callbacks used during the training.

	Returns:
		history (): The training history of the model
	"""
	history = model.fit(train_features,
			  						  train_labels,
			  							batch_size=batch_size,
			  							epochs=epochs,
			  							validation_data=(val_features, val_labels),
			  							callbacks=callbacks,
			  							verbose=1)
	
	return history


def plot_model_history(
	history):
	"""
	Plots the history of the model in a graph

	Input:
		history (): The training history of the model

	Returns:
		None
	"""
	results = pd.DataFrame(history.history)
	results.plot(figsize=(8,5))
	plt.grid(True)
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.show()


def store_model(
	model,
	model_name,
	description):
	"""
	Stores the model in the DATA_FOLDER with the name with a description in the neural network descriptions file
	"""	
	return None
