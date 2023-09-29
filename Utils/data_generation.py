import numpy as np
import random

from constants import *
from surfaces import evaluate_zs_from_surface


def polar_samples_unit_circle_for_data_generation(
    n_circles=5, 
    n_samples=20):
    """
    This function samples radiuses and angles from the unit disk to create random surfaces
    
    Input:
        n_circles (int): Optional. The number of concentric circles where the points will be sampled from
        n_samples (int): Optional. The number of samples of the outer circle of radius one. For the rest the number
                        of samples is proportional to the radius.
                        
    Returns:
        rho_samples (np.array): The array containing the radius polar coordinates of the points
        varphi_samples (np.array): The array containing the angle polar coordinates of the points
    """
    
    # Create empty arrays for the samples
    rho_samples = np.empty([0])
    varphi_samples = np.empty([0])
    
    # Sample points from each concentric circle
    for i in range(1, n_circles+1):
        # Each circle has a number of sample proportional to its radius
        n_subsamples = round((1 - (n_circles - i)/n_circles) * n_samples)
        
        # Sample radius and angles (radius is always the same)
        i_radiuses = np.full((n_subsamples,), i/n_circles)
        i_thetas = np.arange(0, 2* np.pi, 2* np.pi/n_subsamples)
        
        # Append the circle samples to the global samples
        rho_samples = np.concatenate((rho_samples, i_radiuses))
        varphi_samples = np.concatenate((varphi_samples, i_thetas))
    
    return rho_samples, varphi_samples



def polar_samples_unit_circle_for_plotting(
    n_radiuses=50, 
    n_angles=50):
    """
    This function samples radiuses and angles from the unit disk
    
    Input:
        n_radiuses (int): Optional. The number of radiuses to create the mesh
        n_samples (int): Optional. The number of angles to create the mesh
                        
    Returns:
        rho_samples (np.array): The array containing the radius polar coordinates of the points
        varphi_samples (np.array): The array containing the angle polar coordinates of the points
    """
    
    # Create two arrays with equidistant points
    rho_samples = np.linspace(0, 1, n_radiuses)
    varphi_samples = np.linspace(0, 2*np.pi, n_angles)
    return rho_samples, varphi_samples


def cartesian_samples_unit_square_for_data_generation(
    n_rows=10,
    n_cols=10):
    """
    Samples points from the unit square

    Input:
        n_rows (int): Optional. The number of samples in the x axis (rows)
        n_cols (int): Optional. The number of samples in the y axis (columns)

    Returns:
        x_samples (np.array). The array containing the x coordinates of the points
        y_samples (np.array). The array containing the y coordinates of the points
    """
    x_samples = np.linspace(-0.95, 0.95, n_rows)
    y_samples = np.linspace(-0.95, 0.95, n_cols)

    return x_samples, y_samples


def from_cartesian_to_polar_coordinates(
    x_coordinates,
    y_coordinates):
    """
    Transform the cartesian coordinates of points to polar coordinates. The function does not make all possible coordinates combinations

    Input:
        x_coordinates (list): The list of cartesian coordinates in the x axes
        y_coordinates (list): The list of cartesin coordinates in the y axes

    Returns:
        rho_coordinates (np.array): The list of rho (radius) coordinates of the points
        varphi_coordinates (np.array): The list of varphi (radians) coordinates of the points
    """
    np_x_coords = np.array(x_coordinates)
    np_y_coords = np.array(y_coordinates)

    rho_coordinates = np.sqrt(np_x_coords**2 + np_y_coords**2)
    varphi_coordinates = np.arctan2(np_y_coords, np_x_coords)
    
    return rho_coordinates, varphi_coordinates
    

def get_random_zernike_coefficients():
    """
    Describes a random surface based on the zernike polynomials returning their indexes and coefficients to plot them
    Input:
        None
        
    Returns:
        zernike_polynomials (list): A list of tuples with zernike polynomials info (m_index, n_index, coefficient)
    """
    
    zernike_polynomials = []
    
    # Create the zernike tuples containing m index, n index and coefficient of the polynomial
    for zpi in ZERNIKE_POLYNOMIALS_INDEXES:

        # Get a random coefficient for the zernike polynomial
        coef = random.uniform(COEFFICIENT_MIN, COEFFICIENT_MAX)
        
        # Create the combined tuple
        zp = zpi + (coef,)
        
        # Append it to the list of zernike polynomials with coefficients
        zernike_polynomials.append(zp)
        
    return zernike_polynomials


def generate_zernike_polynomial_tuples_from_coefficients(
    zernike_coefficients):
    """
    Generates the zernike polynomial tuples with the coefficient of each one of them from a list of zernike coefficients

    Input:
        zernike_coefficients (list):

    Returns:
        zernike_polynomials (list): A list of tuples with zernike polynomials info (m_index, n_index, coefficient)

    """
    zernike_polynomials = []
    
    # Create the zernike tuples containing m index, n index and coefficient of the polynomial
    for zpi, zpc in zip(ZERNIKE_POLYNOMIALS_INDEXES, zernike_coefficients):
        
        # Create the combined tuple
        zp = zpi + (zpc,)
        
        # Append it to the list of zernike polynomials with coefficients
        zernike_polynomials.append(zp)
        
    return zernike_polynomials


def generate_data_for_training(
    n_data,
    features_file_path,
    labels_file_path,
    verbose=False):
    """
    Generates a dataframe with data for training
    
    Input:
        n_data (int): Number of data points to train (A data point consisting on a surface and its zernike coefficients)
        features_file_path (string): The path where the numpy array with the surface z values will be stored
        labels_file_path (string): The path where the numpy array with the coefficients of the surface will be stored
        verbose (bool): Optional. True if more verbosity for errors
        
    Returns:
        None
    """
    
    surface_list = []
    coefficient_list = []
    
    for i in range(0, n_data):
        # Get the zernike polynomials with their coefficients
        zernike_polynomials = get_random_zernike_coefficients()
        
        # Get the points polar coordinates to sample from the surface created by the zernike polynomials
        rho_coordinates, varphi_coordinates = polar_samples_unit_circle_for_data_generation()
            
        # Compute the values of the surface at the points
        success, surface_values = evaluate_zs_from_surface(rho_coordinates,
                                                           varphi_coordinates,
                                                           zernike_polynomials,
                                                           verbose=verbose)
        
        if not success:
            return False, None

        # Append the surface values to the surface list
        surface_list.append(surface_values)
        
        # Get last element (zernike coefficient) of the zernike polynomials tuples to store them in the dataframe
        coefficients = list(map(lambda x: x[-1], zernike_polynomials))
        
        # Convert the list to a numpy array
        np_coefficients = np.array(coefficients)
        
        # Append the surface zernike coefficients to the list of coefficients
        coefficient_list.append(np_coefficients)
    
    # Create the dataframe and add the data
    np.save(features_file_path, surface_list)
    np.save(labels_file_path, coefficient_list)
    return None
