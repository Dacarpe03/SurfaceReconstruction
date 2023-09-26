from constants import *


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