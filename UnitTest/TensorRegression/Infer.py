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
    
#    # Use this to upgrade old models.
#    layersInfo = ('IMG_S32,CONV_K3_O64_Ps:ReLU,' +
#                  'CONV_K3_O32_Ps:ReLU:MP_K2_S2_Ps,' +
#                  'CONV_K3_O32_Ps:ReLU,' +
#                  'CONV_K3_O16_Ps:ReLU:MP_K2_S2_Ps:UP_S2;' +
#                  'CONV_K3_O64_Ps:ReLU,' +
#                  'CONV_K3_O32_Ps:ReLU:UP_S2,' +
#                  'CONV_K3_O32_Ps:ReLU,' +
#                  'CONV_K3_O16_Ps:ReLU,' +
#                  'CONV_K3_O3_Ps:ReLU,REG_S32/32/3')
#    model = Model.makeFromFile(options.get('model'), layersInfo=layersInfo,
#                               testDs=testDs,
#                               gpus=options.get('gpus'))
#    model.initSession()
#    model.save('Models/End2EndNew.fbm')
#    exit(0)
    
    model = Model.makeFromFile(options.get('model'),
                               testDs=testDs,
                               gpus=options.get('gpus'))

    model.printLayersInfo()
    model.initSession()
    print('')

    results = testDs.evaluateModel(model)

    # Inferring one by one:
    predictions, actuals = [], []
    myPrint('\nTesting one by one inference (%d towers) ... '%(len(model.towers)))
    for b, (batchSamples, batchLabels) in enumerate(testDs.batches(1)):
        myPrint('\r  Processing sample %d ... '%(b+1), False)
        output = model.inferOne( batchSamples[0] )
        assert output.shape == batchLabels[0].shape
        predictions += [output]
        actuals += [batchLabels[0]]
       
    predictions, actuals = np.array(predictions), np.array(actuals)
    mse = np.square(np.array(predictions)-actuals).reshape(testDs.numSamples,-1).mean(1).mean()
    myPrint('Done. MSE: %f'%(mse))

    if abs(mse - results['mse'])<.0001:
        myPrint('  Success! (MSEs match)', color='green')
    else:
        myPrint('  Error: The MSEs did not match!\n', color='red')
        exit(1)
