import matplotlib.pyplot as plt
import numpy as np

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
        
    # Create a list with the zernike meshes
    zernike_meshes = []
    
    # Calculate each the zernike mesh
    for m_index, n_index, coefficient in zernike_polynomials:
        success, z_mesh = compute_zernike_polynomial_for_meshgrid(m_index,
                                                                  n_index,
                                                                  rho_mesh,
                                                                  varphi_mesh,
                                                                  verbose=verbose)

        if not success:
            if verbose:
                msg = "Unable to compute the surface"
                print_verbose_msg(msg, 
                                  ERROR)
            return None

        # Weight the zernike mesh with its coefficient
        weighted_z_mesh = coefficient * z_mesh
        
        # Add the weighted zernike mesh to the meshes list
        zernike_meshes.append(weighted_z_mesh)
        
    # Get the shape of the zernike matrix mesh
    rows = rho_mesh.shape[0]
    columns = rho_mesh.shape[1]
        
    # Create an empty surface mesh
    surface_mesh = np.zeros((rows, columns))
        
    # Combine all the zernike meshes
    for submesh in zernike_meshes:
        surface_mesh += submesh
        
    # Converto to cartesian coordinates
    X, Y = rho_mesh*np.cos(varphi_mesh), rho_mesh*np.sin(varphi_mesh)
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Plot the surface
    ax.plot_surface(X, Y, surface_mesh, cmap=plt.cm.YlGnBu_r)
    plt.show()
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