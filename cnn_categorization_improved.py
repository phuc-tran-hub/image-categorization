from torch import nn

def cnn_categorization_improved(netspec_opts):
    """
     Constructs a network for the improved categorization model.

     Arguments
     --------
     netspec_opts: (dictionary), the improved network's architecture.

     Returns
     -------
     A categorization model which can be trained by PyTorch

    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()
    name = netspec_opts['name']
    kernel_size = netspec_opts['kernel_size']
    num_filters = netspec_opts['num_filters']
    stride = netspec_opts['stride']
    layer = netspec_opts['layer']
    padding = netspec_opts['padding']
    in_channels = 3 
    # add layers as specified in netspec_opts to the network
    for index, layer in enumerate(netspec_opts['layer']):
        if layer == 'conv':              
            net.add_module(name[index],
                           nn.Conv2d(in_channels, num_filters[index], kernel_size[index], stride[index], padding[index]))
        elif layer == 'bn':
            net.add_module(name[index], nn.BatchNorm2d(in_channels))
        elif layer == 'relu':
            net.add_module(name[index], nn.ReLU())
        elif layer == 'pool':
            net.add_module(name[index], nn.AvgPool2d(kernel_size[index], stride[index], padding[index]))
            
        if num_filters[index] != 0:
            in_channels = num_filters[index]
    return net
