import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from constants import MODELS_FOLDER_PATH, \
											DATA_FOLDER_PATH, \
											MODELS_DESCRIPTION_FILE_PATH, \
											KERAS_SUFFIX, \
											NUMPY_SUFFIX


# Importing required libraries
from keras.models import Sequential


def create_conv_model(input_shape,
											output_size):

	# Creating a Sequential model
	model = Sequential()
	input_shape = input_shape + (1, )
	# Adding a convolutional layer with 32 filters, a 3x3 kernel, and 'relu' activation function
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

	# Adding a max-pooling layer with a 2x2 pool size
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Adding another convolutional layer with 64 filters, a 3x3 kernel, and 'relu' activation function
	model.add(Conv2D(64, (3, 3), activation='relu'))

	# Adding another max-pooling layer with a 2x2 pool size
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Flattening the 3D output to 1D tensor for a fully connected layer
	model.add(Flatten())

	# Adding a fully connected layer with 128 units and 'relu' activation function
	model.add(Dense(128, activation='relu'))

	# Adding the output layer with 10 units (for example, for 10 classes) and 'softmax' activation function
	model.add(Dense(output_size, activation='linear'))
	
	# Printing the summary of the model architecture
	model.summary()
	return model


def read_data_for_training(
	features_filename,
	labels_filename,
	train_perc=0.8,
	val_perc=0.1,
	test_perc=0.1):
	"""
	Function to read and split our data given a name (NO PATH NOR SUFFIX, JUST THE NAME)

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

	# Create the path to the data files
	features_file_path = f"{DATA_FOLDER_PATH}/{features_filename}{NUMPY_SUFFIX}"
	labels_file_path = f"{DATA_FOLDER_PATH}/{labels_filename}{NUMPY_SUFFIX}"

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
	name="SurfaceReconstructor"):

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

		# Define layer
		model.add(keras.layers.Dense(neurons,
							kernel_regularizer=regularizer,
							kernel_initializer=keras.initializers.HeNormal(seed=None),
							use_bias=False))

		# Add normalization
		if use_batch_normalization:
			model.add(keras.layers.BatchNormalization())

		# Define the activation function
		model.add(keras.layers.Activation(hidden_activation))

	# Add output layer
	model.add(keras.layers.Dense(output_size,
				activation=output_activation))

	return model


def compile_linear_model(
	model,
	loss_function,
	optimizer,
	metric):
	"""
	Tells the model how to train

	Input:
		model(keras.Sequential): The sequential model to compile
		loss_function (keras.losses): The loss function used to update the weights of the neurons (eg. MeanSquaredError())
		optimizer (keras.optimizers): The optimizer used to update the weights of the neurons (eg. Adam)
		metric (keras.metrics): The metrics to monitor during the training (eg. MeanSquaredError())

	Returns:
		None
	"""
	model.compile(loss=loss_function,
								optimizer=optimizer,
								metrics=[metric])
	return None


def train_linear_model(
	model,
	train_features,
	train_labels,
	batch_size,
	epochs,
	val_features,
	val_labels,
	callbacks):

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

	return None


def store_model(
	model,
	model_name,
	description):
	"""
	Stores the model in the DATA_FOLDER with the name with a description in the neural network descriptions file

	Input:
		model (keras.models): The model to save in the models folder
		model_name (string): The name of the model
		description (string): The description of the model 

	Returns:
		None
	"""
	# Create the model path
	model_file_path = f"{MODELS_FOLDER_PATH}/{model_name}{KERAS_SUFFIX}"
	# Save the model
	model.save(model_file_path)

	# Save its description
	with open(MODELS_DESCRIPTION_FILE_PATH, 'a') as f:
		f.write(f"===={model_name}====\n")
		f.write(description)
		f.write("\n\n")

	return None


def load_model(
	model_name):
	"""
	Loads a model given its name

	Input:
		model_name (string): The name of the model to load

	Returns:
		model (keras.models): The loaded model
	"""
	model_path = f"{MODELS_FOLDER_PATH}/{model_name}{KERAS_SUFFIX}"
	model = keras.models.load_model(model_path)
	return model


def predict_zernike_coefficients(
	model,
	surface_points):
	"""
	Uses the model to get the coefficients of a surface given its samples

	Input:
		model (keras.models): The model that will make the prediction
		surface_points (np.array): The array containing the z_values of the surface in the sampled points

	Returns:
		zernike_coefficients (np.array): The array containing the predicted zernike coefficients that describe the surface
	"""

	# Adjust the surface points to the neural network/model input 
	input_surface = np.array([surface_points])

	# Predict
	output_coefficients = model.predict(input_surface)

	# Get the coefficients compile_linear_model
	zernike_coefficients = output_coefficients[0]

	return zernike_coefficients


def evaluate_model(
	model,
	features,
	labels
	):
	"""
	Evaluates a model and prints its mean squared error
	
	Input:
		model (keras.models): The model to evaluate
		features (np.array): The array with the features to predict
		labels (np.array): The labels of the features to evaluate

	Returns:
		None
	"""
	results = model.evaluate(features, labels)
	print("MSE:", results[1])


def custom_evaluation(
	model,
	features,
	true_labels):
	"""
	Computes the root mean squared error

	Input:
		model (keras.models): The model to evaluate
		features (np.array): The array with the features to predict
		pred_labels (np.array): The array with the predicted labels

	Returns:
		None
	"""

	pred_labels = model.predict(features)
	n_labels = len(pred_labels)
	n_coeffs = len(pred_labels[0])
	print(n_labels)
	squared_differences = np.zeros((n_coeffs))
	for i in range(n_labels):
		squared_differences += np.square(np.subtract(true_labels[i], pred_labels[i]))
	
	squared_differences /= n_labels
	rmse = np.sqrt(squared_differences)
	print(rmse)
	return None