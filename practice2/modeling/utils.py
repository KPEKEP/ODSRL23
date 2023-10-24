def check_same_device(module):
    """
    Check if all parameters and sub-modules of the given module are on the same device.
    
    Args:
    - module (torch.nn.Module): The module to check.

    Returns:
    - bool: True if all are on the same device, False otherwise.
    """
    
    parameters = list(module.parameters())
    
    # If the module doesn't have any parameters, just check its children
    if len(parameters) == 0:
        for child in module.children():
            if not check_same_device(child):
                return False
        return True

    # Get the device of the first parameter
    main_device = parameters[0].device

    # Check if all parameters are on the main device
    for param in parameters:
        if param.device != main_device:
            return False

    # Recursively check if all sub-modules are on the main device
    for child in module.children():
        if not check_same_device(child):
            return False

    return True
