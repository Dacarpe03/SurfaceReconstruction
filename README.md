# SurfaceReconstruction

## Installation (Ubuntu)
From a terminal in the root folder of the project, create the conda environment using the YAML file:
```
conda env create -f environment.yml
```

Now let Python to use the Utils scripts with the following command:
```
export PYTHONPATH="${PYTHONPATH}:./SurfaceReconstruction/Utils/"
```

## Step by step

### 1. Create the data
With Jupyter Notebooks, run the [SurfaceZernikeModesCreation.ipynb](SurfaceZernikeModesCreation.ipynb) notebook

### 2. Train a neural network
With Jupyter Notebooks, run the [TrainNeuralNetwork.ipynb](TrainNeuralNetwork.ipynb) notebook

### 3. Evaluate the model
With Jupyter Notebooks, run the [ModelEvaluation.ipynb](ModelEvaluation.ipynb) notebook

## Results
Considering that the zernike coefficients used to create our surfaces are in the range of [-10, 10] the models are not very accurate
### Linear Model
For a simple linear model with 3 hidden layers of 512, 256 and 64 neurons we have a mean squared error in the test dataset of 8.1
![Linear Model Reconstruction](https://github.com/Dacarpe03/SurfaceReconstruction/blob/master/Images/linear_network.png?raw=true)

### Big Linear Model
For a bigger linear model with 5 hidden layers of  1024, 512, 256, 64 and 32 neurons we have a mean squared error in the test dataset of 10.73

![Big Linear Model Reconstruction](https://github.com/Dacarpe03/SurfaceReconstruction/blob/master/Images/big_linear_network.png?raw=true)

### Convolutional Model
For a convolutional model with 2 layers of convolution with 32 and 64 neurons and 1 hidden layers of 128 neurons  we have a mean squared error in the test dataset of 5.57
![Convolutional Model Reconstruction](https://github.com/Dacarpe03/SurfaceReconstruction/blob/master/Images/convolutional_network.png?raw=true)
