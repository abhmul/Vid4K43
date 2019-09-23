def get_sample_shape(inputs):
    """Gets the shape of the input excluding the batch size"""
    return tuple(inputs.size())[1:]


def get_num_channels(inputs):
    return inputs.size(1)


def get_image_shape(inputs):
    return tuple(inputs.size())[2:]


def flatten(x):
    return torch.view(x.size(0), -1)
