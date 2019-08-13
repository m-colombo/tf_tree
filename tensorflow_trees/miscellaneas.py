
def interpolate_layers_size(input_size: int, output_size: int, layers: int, fatness: float = 0, skewness: float = 0):
    """ Currently only linear interpolation is supported (i.e. fatness = skewness = 0)

    :param input_size:
    :param output_size:
    :param layers:
    :param fatness:
        > 0 intermediate layers are bigger than linear interpolation
        < 0 intermediate layers are smaller than linear interpolation
        = 0 linear interpolation
    :param skewness:
        > 0 bigger interpolation is moved towards the final layers
        > 0 bigger interpolation is moved towards the initial layers
        = 0 central
    :return:
    """
    if fatness != 0:    # TODO
        raise NotImplemented()

    if skewness != 0:   # TODO
        raise NotImplemented()

    # linear
    for i in range(layers):
        yield int(input_size + (output_size - input_size) * (i + 1) / (layers))
