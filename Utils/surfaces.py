import numpy as np
import math

from constants import *


def print_verbose_msg(msg,
                      level):
    """
    This function prints a message and its level of severity
    
    Input:
        msg (string): The message to display
        level (int): The dictionary key that contains the level of severity
    
    Returns:
        None
    """
    
    complete_msg = f"{VERBOSITY_DICT[level]}: {msg}"
    print(complete_msg)
    return None
 

def check_zernike_parameters(
    m_index,
    n_index,
    rho,
    verbose=False):
    """
    This function checks that the zernike parameters are valid    
    Input:
        m_index (int): The m number of the Zernike Polynomial
        n_index (int): The n number of the Zernike Polynomial
        rho (float): The distance of the point to the center (radius in polar coordinates) 
        verbose (bool): Indicates if the errors are displayed in console. Default is False.
        
    Returns:
        bool: True if the parameters are correct, False otherwise
    """

    # If n is lower than m there is an error
    if n_index < abs(m_index):
        if verbose:
            msg = f"The n index is smaller than m index: {n_index} < {abs(m_index)}"
            print_verbose_msg(msg, ERROR)
        return False
        
    # If the radius is less than 0 or greater than 1 there is an error
    if rho < 0 or rho > 1:
        if verbose:
            msg = f"The radius (ro) must be between -1 and 1. Radius is {rho}"
            print_verbose_msg(msg, ERROR)
        return False


    # If everything is correct, return True
    return True


def compute_radial_value(
    m_index,
    n_index,
    rho, 
    verbose=False):
    """
    Computes the radial polynomial at ro given m and n indexes of the Zernike Polynomials
        
    Input:
        m_index (int): The m number of the Zernike Polynomial
        n_index (int): The n number of the Zernike Polynomial
        rho (float): The distance of the point to the center (radius in polar coordinates)
        verbose (bool): If True, error and warning messages will show in the terminal
            
    Returns:
        r_value(float): The value of the radial polynomial at rho
    """
        
    # Default value of the polynomial
    if rho == 1:
        return 1
    
    # Value if n-m is odd
    r_value = 0
        
    n_minus_m = n_index - m_index
    module = n_minus_m % 2
        
    # If the module of n-m is even then we compute
    if module == 0:

        # Compute the index of the sumatory and add 1 for the loop
        sumatory_index = (n_minus_m)//2 + 1
            
        # The summatory
        for k in range(0, sumatory_index):
                
            # Compute the numerator: (-1)^k * (n-k)!
            numerator = (-1)**k * math.factorial(n_index-k)
                
            # Compute the numbers from which we will compute their factorials for the denominator
            n_plus_m_d2_minus_k = (n_index + m_index)//2 - k  # (n+m)/2 - k
            n_minus_m_d2_minus_k = (n_index - m_index)//2 - k # (n-m)/2 - k
                
            # Compute denominator: k! * ((n+m)/2 - k)! *((n-m)/2 - k)!
            denominator = math.factorial(k) * math.factorial(n_plus_m_d2_minus_k) * math.factorial(n_minus_m_d2_minus_k)
                
            # Update the total sum
            r_value += numerator/denominator * rho**(n_index-2*k)
            
    return r_value


def compute_zernike_polynomial_at_point(
    m_index,
    n_index,
    rho,
    varphi,
    verbose=False):
    """
    This function computes the value of a Zernike polynomial on polar coordinates given the Zernike coefficients
        
    Input:
        m_index (int): The m number of the Zernike Polynomial
        n_index (int): The n number of the Zernike Polynomial
        rho (float): The distance of the point to the center (radius in polar coordinates) 
        varphi (float): The angle of the point with respect to x-axis (angle in polar coordinates)
        verbose (bool): If True, error and warning messages will show in the terminal
            
    Returns:
        success (bool): True if the computing did not fail
        zernike_value (float): The value of the zernike_polynomial at the point
    """
        
    # First check if the zernike parameters are valid
    success = check_zernike_parameters(m_index, 
                                       n_index, 
                                       rho, 
                                       verbose=verbose)

    if not success:
        return success, None
        
    # Obtain the absolute value of the m index
    m_abs_index = abs(m_index)
        
    # Obtain the sign of the m index
    m_sign = 1
    if m_abs_index > 0:
        m_sign = m_index // m_abs_index
    
    # Compute the radial polynomial value at rho
    r_value = compute_radial_value(m_abs_index, 
                                   n_index, 
                                   rho,
                                   verbose=verbose)
        
    # Compute the zernike value
    if m_sign == 1:
        zernike_value = r_value * math.cos(m_abs_index * varphi)
    else:
        zernike_value = r_value * math.sin(m_abs_index * varphi)
            
    return success, zernike_value


def compute_surface_value_at_point(
    rho,
    varphi,
    zernike_polynomials,
    verbose=False):
    """
    Function to compute the z value of a surface at a point given zernike coefficients and polar coordinates of point
        
    Input:
        rho (float): The distance of the point to the center (radius in polar coordinates) 
        varphi(float): The angle of the point with respect to x-axis (angle in polar coordinates)
        zernike_polynomials (list): A list of tuples with zernike polynomials info (m_index, n_index, coefficient)
        verbose (bool): Optional. True if more verbosity for errors
        
    Returns:
        success (bool): True if the computation had no failures
        z_value (float): The value of the surface at the point
    """
        
    # Initialize the z value of the point
    z_value = 0


    # For points outside of the unit disk we return the value 0 (this is necessary for creating the convolution data)
    if rho < 0 or rho > 1:
        return True, z_value
        
    # For loop to compute the point value on each zernike polynomials and add them all with a weighted sum
    for m_index, n_index, coefficient in zernike_polynomials:
        success, actual_value = compute_zernike_polynomial_at_point(m_index,
                                                                    n_index,
                                                                    rho,
                                                                    varphi,
                                                                    verbose=verbose)

        # If there was an error then return
        if not success:
            return success, None

        # Add the weighted value
        weighted_value = coefficient * actual_value
        z_value += weighted_value

        
    return success, z_value


def evaluate_zs_from_surface(
    rho_coordinates,
    varphi_coordinates,
    zernike_polynomials,
    verbose=False):
    """
    Evaluates the z values of a surface given its zernike polynomials and polar coordinates of a set of points
    
    Input:
        rho_coordinates (np.array): The array containing the radius polar coordinates of the points
        varphi_coordinates (np.array): The array containing the angle polar coordinates of the points
        zernike_polynomials (list): A list of tuples with zernike polynomials info (m_index, n_index, coefficient)
        verbose (bool): Optional. If true, more verbosity for errors

    Returns:
        success (bool): True if there were no errors during computation
        z_values (np.array): The array containing the z value of the surface in the coordinates
    """
    
    # The list to store the z values
    z_list = []
    
    # Loop to compute the z value on all the given points
    for (rho, varphi) in zip(rho_coordinates, varphi_coordinates):
        success, z_value = compute_surface_value_at_point(rho,
                                                          varphi,
                                                          zernike_polynomials,
                                                          verbose=verbose)

        if not success:
            return success, None
        
        z_list.append(z_value)
    
    # Convert list to numpy array
    z_values = np.array(z_list)
    
    return success, z_values


def compute_zernike_polynomial_for_meshgrid(m_index,
                                            n_index, 
                                            rho_mesh,
                                            varphi_mesh,
                                            verbose=False):
    """
    Function to compute the zernike polynomials z component for plotting
    
    Input:
        m_index (int): The m number of the Zernike Polynomial
        n_index (int): The n number of the Zernike Polynomial
        rho_mesh (np.array): A 2d array containing the radius of the points' polar coordinates
        varphi_mesh(np.array): A 2d array containing the angle of the points' polar coordinates
        verbose (bool): Optional. True if more verbosity for errors
        
    Returns:
        success (bool): True if the computation succeeded
        zernike_mesh: The values of the zernike polynomial in the mesh grid
    """
    
    # Get rows and columns from mesh to create the Z axis matrix mesh
    rows = rho_mesh.shape[0]
    columns = rho_mesh.shape[1]
    
    # Create an empty 2d array
    zernike_mesh = np.empty((rows, columns))
    
    # Fill the Z data array with the zernike values of the points
    for r in range(0, rows):
        for c in range(0, columns):
            # Get rho and varphi from the meshes
            rho = rho_mesh[r][c]
            varphi = varphi_mesh[r][c]
            
            # Assign the result in the Z axis matrix mesh
            success, value = compute_zernike_polynomial_at_point(m_index,
                                                                 n_index,
                                                                 rho,
                                                                 varphi,
                                                                 verbose=verbose)

            if not success:
                return False, None

            # Assign the result in the Z axis matrix mesh
            zernike_mesh[r][c] = value
            
    return True, zernike_mesh


def compute_surface_meshgrid(
    zernike_polynomials,
    rho_mesh,
    varphi_mesh,
    verbose=False):
    """
    Computes the surface z values given rho and varphi meshes

    Input:
        zernike_polynomials (list): A list of tuples with zernike polynomials info (m_index, n_index, coefficient)
        rho_mesh (np.array): A 2d array containing the radius of the points' polar coordinates
        varphi_mesh(np.array): A 2d array containing the angle of the points' polar coordinates
        verbose (bool): Optional. True if more verbosity for errors
    """

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

    return surface_mesh


def convert_list_from_cartesian_to_polar_coordinates(
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


def convert_point_from_cartesian_to_polar_coordinates(
    x_coordinate,
    y_coordinate):
    """
    Convert the point from cartesian 

    Input:
        x_coordinate (float): The x coordinate of the point to transfrom
        y_coordinate (float): The y coordinate of the point to transform


    Returns:
        rho (float): The rho coordinate of the point transformed to polar
        varphi (float): The varphi coordinate of the point transformed to polar
    """
    
    rho = math.sqrt(x_coordinate**2 + y_coordinate**2)
    varphi = math.atan2(y_coordinate, x_coordinate)

    return rho, varphi