import numpy as np
import time
from fireball import Model, DashOptions, myPrint
from fireball.datasets.cifar import CifarDSet

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
 
    model = Model.makeFromFile(options.get('in'))
    model.printLayersInfo()
    model.initSession()
    print('')

    # Note that CIFAR images are in RGB format
    model.exportToCoreMl(options.get('out'), isBgr=False)
  
    coreMlModel = coremltools.models.MLModel(options.get('out'))
    # Print CoreML Model Spec
    #coremltools.models.neural_network.printer.print_network_spec(coreMlModel.get_spec())

    # Initialize the dataset:
    myPrint('\nPreparing CIFAR-100 dataset ... ', False)
    testDs = CiFarAutoDSet.makeDatasets('test', 128)
    myPrint('Done.')

    myPrint('\nEvaluate Original model ... ')
    resultsOrg = model.evaluateDSet(testDs)

    myPrint('\nEvaluate the exported CoreML Model ... ')
    predictions = []
    actuals = []
    t0 = time.time()
    for b, (batchSamples, batchLabels) in enumerate(testDs.batches(1)):
        myPrint('\r  Processing sample %d ... '%(b+1), False)
        imgNp = batchSamples[0]
        img = PIL.Image.fromarray(imgNp)
#        img.show()
        output_dict = coreMlModel.predict({'input': img}, useCPUOnly=True)
        predictions += [ output_dict['output'].reshape(3,32,32).transpose(1,2,0) ]
        actuals += [ batchLabels[0] ]

    myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)           \n'%(testDs.numSamples, time.time()-t0))
    resultsCoreML = testDs.evaluate(np.array(predictions), np.array(actuals))
    
    if abs(resultsCoreML['psnr'] - resultsOrg['psnr'])<.001:
        myPrint('    Success! (Accuracies match)', color='green')
    else:
        myPrint('\n    Error: The accuracies do not match between the original model and exported model!\n', color='red')
        exit(1)

    if options.get('netron'):
        myPrint('\n\nVisualizing the network using "Netron". Press Ctrl+C to exit.')
        import netron
        netron.start(options.get('out'))
