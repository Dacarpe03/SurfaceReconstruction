from constants import *


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