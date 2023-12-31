import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


from surfaces import compute_surface_meshgrid, \
                     convert_list_from_cartesian_to_polar_coordinates

from data_generation import polar_samples_unit_circle_for_plotting, \
                            generate_zernike_polynomial_tuples_from_coefficients


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


def plot_surface_enhanced(
    zernike_polynomials,
    n_radiuses=50,
    n_angles=50,
    verbose=False):
    """
    Plots a surface given the zernike polynomials and its coefficients using plotly
    
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
    
    # Convert to cartesian coordinates
    X, Y = rho_mesh*np.cos(varphi_mesh), rho_mesh*np.sin(varphi_mesh)

    
    # Create figure
    fig = go.Figure(data=[go.Surface(z=surface_mesh, x=X, y=Y), go.Surface(z=surface_mesh, x=Y, y=X)])
    fig.update_layout(title='Surface',
                      autosize=True,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()
    return None


def plot_surface_from_zernike_coefficients(
    zernike_coefficients,
    enhanced=True,
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

    if enhanced:
        plot_surface_enhanced(zernike_polynomials,
                              verbose=verbose)
    else:
        # Then plot the surface
        plot_surface(zernike_polynomials, 
                     verbose=verbose)

    return None


def plot_original_vs_reconstructed(
    original_zernike_coeffs,
    predicted_zernike_coeffs,
    name,
    n_radiuses=50,
    n_angles=50,
    verbose=False):
    
    # Build the zernike polynomials from the coefficients
    original_zernike_polynomials = generate_zernike_polynomial_tuples_from_coefficients(original_zernike_coeffs)
    predicted_zernike_polynomials = generate_zernike_polynomial_tuples_from_coefficients(predicted_zernike_coeffs)

    # Get the radius and angle samples
    rho_samples, varphi_samples = polar_samples_unit_circle_for_plotting(n_radiuses=n_radiuses,
                                                                         n_angles=n_angles)
    # Create the mesh grid from the sampled data
    rho_mesh, varphi_mesh = np.meshgrid(rho_samples, varphi_samples)

    # Compute the original surface mesh
    original_surface_mesh = compute_surface_meshgrid(original_zernike_polynomials,
                                                     rho_mesh,
                                                     varphi_mesh,
                                                     verbose=verbose)

    # Compute reconstructed surface_mesh
    predict_surface_mesh = compute_surface_meshgrid(predicted_zernike_polynomials,
                                                    rho_mesh,
                                                    varphi_mesh,
                                                    verbose=verbose)

    # Convert to cartesian coordinates
    X, Y = rho_mesh*np.cos(varphi_mesh), rho_mesh*np.sin(varphi_mesh)

    
    # Create Surfaces
    og_surface = go.Surface(z=original_surface_mesh, x=X, y=Y, colorscale='viridis', colorbar_x=-0.2, colorbar=dict(len=0.5))
    ai_surface = go.Surface(z=predict_surface_mesh, x=X, y=Y, colorbar=dict(len=0.5))

    og_surface_vs = go.Surface(z=original_surface_mesh, x=X, y=Y, colorscale='viridis', colorbar_x=-0.2, colorbar=dict(len=0.5))
    ai_surface_vs = go.Surface(z=predict_surface_mesh, x=X, y=Y, colorbar=dict(len=0.5))

    fig = make_subplots(rows=2, cols=2,
                        horizontal_spacing=0.05,
                        vertical_spacing=0.05,
                        specs=[[{'is_3d': True, 'type':'scene'}, {'is_3d': True, 'type':'scene'}],
                               [{'is_3d': True, 'colspan': 2, 'type':'scene'}, None]],
                        subplot_titles=['Original surface', 'AI surface', 'Original vs AI', None])

    fig.add_trace(og_surface_vs, 2, 1)
    fig.add_trace(ai_surface_vs, 2, 1)
    fig.add_trace(og_surface, 1, 1)
    fig.add_trace(ai_surface, 1 ,2)
    fig.update_layout(title_text=f"Reconstruction with {name}",
                      height=1000,
                      showlegend=False)
    fig.show()
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


def plot_2d_cartesian_points(
    x_samples,
    y_samples):
    """
    Plots points given its unique x and y coordinates, (the funciton makes a cartesian product to get all the possible combinations)

    Input:
        x_samples (np.array). The array containing the x coordinates of the points
        y_samples (np.array). The array containing the y coordinates of the points

    Returns:
        None
    """
    x_coordinates = []
    y_coordinates = []
    for x in x_samples:
        for y in y_samples:
            x_coordinates.append(x)
            y_coordinates.append(y)


    rho_coordinates, varphi_coordinates = convert_list_from_cartesian_to_polar_coordinates(x_coordinates,
                                                                                           y_coordinates)
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    
    # Remove radius ticks and set the radius to 1
    ax.set_rticks([])
    ax.set_rlim(0, 1)
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
    ax.set_thetagrids(angles, labels=labels)

    # Plot the points
    ax.plot(varphi_coordinates, rho_coordinates, 'k.')