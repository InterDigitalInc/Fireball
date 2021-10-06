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
    ('gpus',       'The GPUs to use. (Default: All available GPUs)',                        ['string', 'default:all'])
])
    
# **********************************************************************************************************************
# Subclassing the original MNIST dataset:
class EvenOddDSet(MnistDSet):
    # ******************************************************************************************************************
    @classmethod
    def postMakeDatasets(cls):
        # This is called at the end of a call to makeDatasets
        EvenOddDSet.numClasses = 2
        EvenOddDSet.classNames = ['Even', 'Odd']

    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        return self.samples[batchIndexes], (self.labels[batchIndexes])%2
    
# **********************************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)
    
    # Initialize the dataset:
    myPrint('\nPreparing Even/Odd dataset ... ')
    testDs = EvenOddDSet.makeDatasets('test', batchSize=128)

    model = Model.makeFromFile(options.get('model'),
                               testDs=testDs,
                               gpus=options.get('gpus'))
        
    model.printLayersInfo()
    model.initSession()
#    Layers.printAllParamInfo(model.layers)
    print('')
    
    # Inferring one by one:
    numMatches = 0
    myPrint('Testing one by one inference (%d towers) ... '%(len(model.towers)))
    for b, (batchSamples, batchLabels) in enumerate(testDs.batches(1)):
        myPrint('\r  Processing sample %d ... '%(b+1), False)
        prob = model.inferOne( batchSamples[0], returnProbs=True )
        label = model.inferOne( batchSamples[0], returnProbs=False )
        
        assert label == np.int32(np.round(prob))
        numMatches += 1 if batchLabels[0] == label else 0
        
    myPrint('Done.')
    accuracy = (100.0*numMatches)/testDs.numSamples
    accuracyFromModel = 100.0-model.evaluateDSet(testDs, quiet=True, returnMetric=True)
    assert np.allclose(accuracy, accuracyFromModel), "Accuracies don't match: %f != %f"%(accuracy, accuracyFromModel)
    myPrint( "Accuracy(%%): %f"%(accuracy))
