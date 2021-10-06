import numpy as np
import time
from fireball import Model, DashOptions, myPrint
from fireball.datasets.mnist import MnistDSet

import PIL
import coremltools

# **********************************************************************************************************************
options = DashOptions(
[
    ('help',        'Show this help',                                           ['bool']),
    ('h',           None,                                                       ['bool']),

    ('in',          'Path to the model file',                                   ['string', 'required']),
    ('out',         'Path to the CoreML file',                                  ['string', 'required']),
    ('netron',      'Visualize the network after the export using "Netron".',   ['bool']),
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
 
    model = Model.makeFromFile(options.get('in'))
    model.printLayersInfo()
    model.initSession()
    print('')

    model.exportToCoreMl(options.get('out'), rgbBias=-.5, scale=1.0/255)

    coreMlModel = coremltools.models.MLModel(options.get('out'))

    # Initialize the dataset:
    myPrint('\nPreparing MNIST datasets for regression... ')
    testDs = RegMnistDSet.makeDatasets('test', 128)
    myPrint('Done.')

    myPrint('\nEvaluate Original model ... ')
    resultsOrg = model.evaluateDSet(testDs)

    myPrint('\nEvaluate the exported CoreML Model ... ')
    predictions = []
    actuals = []
    t0 = time.time()
    for b, (batchSamples, batchLabels) in enumerate(testDs.batches(1)):
        myPrint('\r  Processing sample %d ... '%(b+1), False)
        imgNp = np.uint8((np.float32(batchSamples[0])+.5)*255.0).reshape(28,28)
        img = PIL.Image.fromarray(imgNp,'L')
#        img.show()
        output_dict = coreMlModel.predict({'input': img}, useCPUOnly=True)
        output = output_dict['output'].reshape(-1)
        
        predictions += [ output ]
        actuals += [ batchLabels[0] ]

    myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)           \n'%(testDs.numSamples, time.time()-t0))
    resultsCoreML = testDs.evaluate(np.array(predictions), np.array(actuals))

    if abs(resultsCoreML['psnr'] - resultsOrg['psnr'])<.1:
        myPrint('    Success! (Accuracies match)', color='green')
    else:
        myPrint('\n    Error: The accuracies do not match between the original model and exported model!\n', color='red')
        exit(1)
        
    if options.get('netron'):
        myPrint('\n\nVisualizing the network using "Netron". Press Ctrl+C to exit.')
        import netron
        netron.start(options.get('out'))
