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
    
 
def check_zernike_parameters(m_index, 
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
    
    return True
    
    
def compute_zernike_polynomial_at_point(m_index,
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
        
   	Returns:
    	zernike_value (float): The value of the zernike_polynomial at the point
    """
    
    # First check if the zernike parameters are valid
    if not check_zernike_parameters(m_index, 
                                    n_index, 
                                    rho, 
                                    verbose):
    return False, None
    
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
                                   verbose)
    
    # Compute the zernike value
    if m_sign == 1:
    	zernike_value = r_value * math.cos(m_abs_index * varphi)
    else:
    	zernike_value = r_value * math.sin(m_abs_index * varphi)
        
    return zernike_value