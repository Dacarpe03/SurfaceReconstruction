import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from constants import *
from surfaces import *
from data_generation import *


def plot_surface(
    zernike_polynomials,
    n_radiuses=50,
    n_angles=50,
    verbose=False):
    """
    Plots a surface give the zernike polynomials and its coefficients
    
    Input:
        zernike_polynomials (list): A list of tuples with polynomials info (m_index, n_index, coefficient)
        n_radiuses (int): Optional. The number of radiuses to create the mesh
        n_angles (int): Optional. The number of angles to create the mesh
        
    Returns:
        None
    """
        
    # Get the radius and angle samples
    rho_samples, varphi_samples = polar_samples_unit_circle_for_plotting(n_radiuses=n_radiuses,
                                                                         n_angles=n_angles)
    # Create the mesh grid from the sampled data
    rho_mesh, varphi_mesh = np.meshgrid(rho_samples, varphi_samples)

    # Compute the surface mesh
    surface_mesh = compute_surface_meshgrid(zernike_polynomials,
                                            rho_mesh,
                                            varphi_mesh,
                                            verbose=verbose)
        
    # Converto to cartesian coordinates
    X, Y = rho_mesh*np.cos(varphi_mesh), rho_mesh*np.sin(varphi_mesh)
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Plot the surface
    ax.plot_surface(X, Y, surface_mesh, cmap=plt.cm.YlGnBu_r)
    plt.show()
    return None


def plot_surface_from_zernike_coefficients(
    zernike_coefficients,
    verbose=False):
    
    """
    This function plots a surface given its zernike coefficients

    Input:
        zernike_coefficients (list): A list of zernike coefficients

    Returns:
        None
    """

    # First create the zernike polynomial tuples
    zernike_polynomials = generate_zernike_polynomial_tuples_from_coefficients(zernike_coefficients)

    # Then plot the surface
    plot_surface(zernike_polynomials, 
                 verbose=verbose)

    return None


def plot_2d_polar_points(
    rho_samples, 
    varphi_samples):
    """
    Plots a 2d graphic with points given radiuses and thetas
    
    Input:
        rho_samples (np.array): The array containing the radius polar coordinates of the points
        varphi_samples (np.array): The array containing the angle polar coordinates of the points
    
    Returns:
        None
    """
    
    # Create a plot with polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    
    # Remove radius ticks
    ax.set_rticks([])
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    ax.set_thetagrids(angles, labels=labels)

    # Plot the points
    ax.plot(varphi_samples, rho_samples, 'k.')


def test_plot():
    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    z = z_data.values
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
    fig.show()