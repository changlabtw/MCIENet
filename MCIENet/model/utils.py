import torch.nn as nn

def get_activite_func(name, **kwargs):
    """
    This function returns an activation function based on the given name.

    Args:
    - name (str): The name of the activation function to retrieve.
    - **kwargs: Additional keyword arguments that can be passed to specific activation functions.

    Returns:
    - nn.Module or None: The corresponding activation function if the name is valid, None otherwise.
    """
    if name == 'softmax':
        return nn.Softmax() # muti-class
    elif name == 'sigmoids':
        return nn.Sigmoid() # binary
    elif name == 'relu':
        return nn.ReLU() # linear
    elif name == 'gelu':
        return nn.GELU() # Transformer 
    elif name in ['leaky_relu', 'leakyrelu']:
        return nn.LeakyReLU(kwargs['slope']) if 'slope' in kwargs else nn.LeakyReLU() 
    else:
        return None

def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)