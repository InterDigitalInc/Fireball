import numpy as np
import time
from fireball import Model, DashOptions, myPrint
from fireball.layers import Layers
from fireball.datasets.mnist import MnistDSet

# **********************************************************************************************************************
options = DashOptions(
[
    ('help',     'Show this help',                                                          ['bool']),
    ('h',        None,                                                                      ['bool']),

    ('model',      'Path to the model file',                                                ['string', 'required']),
    ('time',       'Calculate Inference time per Sample',                                   ['bool']),
    ('maxSamples', 'Number of samples to use for testing. (Default: All test samples)',     ['int', 'min:1', 'default:0']),
    ('gpus',       'The GPUs to use. (Default: All available GPUs)',                        ['string', 'default:all'])
])
    
    
# **********************************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)
    
    # Initialize the dataset:
    myPrint('\nPreparing MNIST dataset ... ')
    testDs = MnistDSet.makeDatasets('test', batchSize=128)

    model = Model.makeFromFile(options.get('model'),
                               testDs=testDs,
                               gpus=options.get('gpus'))
        
    model.printLayersInfo()
    model.initSession()
#    Layers.printAllParamInfo(model.layers)
    print('')

    maxSamples = options.get('maxSamples')
    if maxSamples==0:   maxSamples = testDs.numSamples

    inferResults = []
    if options.get('time'):
        myPrint('Running inference on %d Test Samples (one by one, using %d towers) ... '%(maxSamples,
                                                                                           len(model.towers)))
        model.evaluateDSet(testDs, batchSize=1, maxSamples=maxSamples)
    else:
        myPrint('Running inference on %d Test Samples (batchSize:%d, %d towers) ... '%(maxSamples,
                                                                                       testDs.batchSize,
                                                                                       len(model.towers)))
        model.evaluateDSet(testDs, maxSamples=maxSamples)

