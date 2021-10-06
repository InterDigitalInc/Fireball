import numpy as np
import time
from fireball import Model, DashOptions, myPrint
from fireball.datasets.cifar import CifarDSet

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
# Subclassing the original CiFar dataset:
class CiFarAutoDSet(CifarDSet):
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=64, coarseLabels=False):
        super().__init__(dsName, dataPath, samples, labels, batchSize, coarseLabels)
        self.evalMetricName = 'MSE'
        self.psnrMax = 255

    # ******************************************************************************************************************
    @classmethod
    def postMakeDatasets(cls):
        # This is called at the end of a call to makeDatasets
        CiFarAutoDSet.numClasses = 0
        
    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        return self.samples[batchIndexes], self.samples[batchIndexes]

# **********************************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)
    
    # Initialize the dataset:
    myPrint('\nPreparing CIFAR-100 dataset ... ')
    testDs = CiFarAutoDSet.makeDatasets('test', batchSize=128)

    model = Model.makeFromFile(options.get('model'),
                               testDs=testDs,
                               gpus=options.get('gpus'))
        
    model.printLayersInfo()
    model.initSession()
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

