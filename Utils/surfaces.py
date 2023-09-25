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


def compute_radial_value(m_index,
                         n_index,
                         rho, 
                         verbose=False):
    """
    Computes the radial polynomial at ro given m and n indexes of the Zernike Polynomials
    
    Input:
        m_index (int): The m number of the Zernike Polynomial
        n_index (int): The n number of the Zernike Polynomial
        rho (float): The distance of the point to the center (radius in polar coordinates)
        
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
    
    # If the module of n-m is even calculate
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


def polar_samples_unit_circle_for_data_generation(n_circles=5, 
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


def polar_samples_unit_circle_for_plotting(n_radiuses=50, 
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


def plot_2d_polar_points(rho_samples, 
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
    labels = ['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    ax.set_thetagrids(angles, labels=labels)
    # Plot the points
    ax.plot(varphi_samples, rho_samples, 'k.')
