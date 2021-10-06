import numpy as np
import time
from fireball import Model, DashOptions, myPrint
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
class RegMnistDSet(MnistDSet):
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=64):
        super().__init__(dsName, dataPath, samples, labels, batchSize)
        self.evalMetricName = 'MSE'

    # ******************************************************************************************************************
    @classmethod
    def postMakeDatasets(cls):
        # This is called at the end of a call to makeDatasets
        RegMnistDSet.numClasses = 0
        
    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        return self.samples[batchIndexes], np.float32(self.labels[batchIndexes]) # Return labels as float32

# **********************************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)
    
    # Initialize the dataset:
    myPrint('\nPreparing MNIST datasets for regression... ')
    testDs = RegMnistDSet.makeDatasets('test', batchSize=128)

    model = Model.makeFromFile(options.get('model'),
                               testDs=testDs,
                               gpus=options.get('gpus'))

    model.printLayersInfo()
    model.initSession()
    print('')
    
    myPrint('\nEvaluate model ... ')
    resultsOrg = model.evaluateDSet(testDs)

    # Inferring one by one:
    predictions, actuals = [], []
    myPrint('\nNow Testing one by one inference (%d towers) ... '%(len(model.towers)))
    t0 = time.time()
    for b, (batchSamples, batchLabels) in enumerate(testDs.batches(1)):
        myPrint('\r  Processing sample %d ... '%(b+1), False)
        output = model.inferOne( batchSamples[0] )
        assert output.shape == batchLabels[0].shape
        predictions += [output]
        actuals += [batchLabels[0]]

    myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)           \n'%(testDs.numSamples, time.time()-t0))
    resultsCoreML = testDs.evaluate(np.array(predictions), np.array(actuals))

    if abs(resultsCoreML['mse'] - resultsOrg['mse'])<.0001:
        myPrint('    Success! (MSEs match)', color='green')
    else:
        myPrint('\n    Error: The accuracies do not match between the original model and exported model!\n', color='red')
        exit(1)
