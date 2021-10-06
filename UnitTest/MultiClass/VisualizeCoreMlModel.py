import numpy as np
import time
from fireball import Model, DashOptions, myPrint

import coremltools

# **********************************************************************************************************************
options = DashOptions(
[
    ('help',        'Show this help',                       ['bool']),
    ('h',           None,                                   ['bool']),

    ('in',          'Path to the model file',               ['string', 'required']),
])

# **********************************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(0)
 
    # Visualization
    import coremltools
    coreMlModel = coremltools.models.MLModel(options.get('in'))
    coremltools.models.neural_network.printer.print_network_spec(coreMlModel.get_spec())
    coreMlModel.visualize_spec(title=options.get('in'))
