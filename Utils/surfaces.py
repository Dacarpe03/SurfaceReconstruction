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
