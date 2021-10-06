import numpy as np
import os

from fireball import Model, DashOptions, myPrint
from fireball.datasets.cifar import CifarDSet

# ************************************************************************************************
options = DashOptions(
    [
        ('help',            'Show this help',                                                   ['bool']),
        ('h',               None,                                                               ['bool']),
     
        ('in',             'Name of original model',                                            ['string', 'required']),
        ('out',            'Name of output (retrained) model',                                  ['string', 'noValDef:']),

        ('batchSize',      'The batch size used in each iteration of training..(Default: 32)',  ['int', 'min:10', 'default:32']),
        ('epochs',         'Number of Epochs for NN algorithm.(Default: 10)',                   ['int', 'min:1', 'default:10']),
        ('regFactor',      'Regularization Factor for NN algorithm.(Default: 0)',               ['float', 'min:0.0', 'default:0']),
        ('dropOut',        'Drop Out Probability for NN algorithm.(Default: 0)',                ['float', 'min:0.0', 'max:0.9', 'default:0']),
        ('learningRate',   'Can be one of the following:(Default: 0.001,0.00001)\n'+
                           'lrValue: Fixed learning rate during the training.\n'+
                           '"(start, end)": learning rate starts at "start" ending in "end".\n'+
                           '"[(1,lr1),(N2,lr2),...,(nN,lrN)]" Piecewise learning rate.',        ['string', 'default:(0.001,0.000001)']),
        ('optimizer',      '"GradientDescent", "Adam", or "Momentum".(Default: Momentum)',      ['string', 'default:Adam']),

        ('gpus',           'The GPUs used for retraining.',                                     ['string']),
        ('restart',        'Delete intermediate files and restart the process from scratch.',   ['bool']),
     
        ('paramSearch',    'Do a grid search on the hyper parameters of the model.',            ['bool']),
        ('numWorkers',     'The number of worker processes for parameter search.(Default: 1)',  ['int', 'default:1']),
        ('log',            'log the parameter search progress to a file.',                      ['bool']),
        ('saveResults',    'Save results to specified csv file.',                               ['string']),
        ('useTest',        'Use test set for evaluation. Otherwise use validation set.',        ['bool']),
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

# ************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)

    if options.get('paramSearch'):
        from fireball.paramsearch import paramSearch
        # Param Names here must match the ones in the command line options defined above.
        parameterChoices = [
                            ('regFactor',    [0.0]),
                            ('dropOut',      [0.1, 0.2]),
                            ('batchSize',    [32]),
                            ('learningRate', ["(0.001,0.000001)", "[(0,0.001),(2000,0.0001),(6000,0.000001)]"]),
                            ('optimizer',    ['Adam']),
                            ('epochs',       [5]),
                           ]

        additionalParams = {'in': options.get('in')}
        
        if options.get('out') is not None:
            additionalParams['out'] = options.get('out')
        
        if options.get('useTest'):
            additionalParams['useTest'] = None

        if options.get('gpus') is not None:
            if options.get('gpus').lower() in ['all', 'half']:
                myPrint('Fatal Error: "gpus" must be a comma delimited list of gpu numbers!', color='red')
                exit(1)
        paramSearch(parameterChoices, additionalParams, nWorkers=options.get('numWorkers'),
                    doLog=options.get('log'), gpus=options.get('gpus'), restart=options.get('restart'))
        exit(0)
    
    outFileName = None
    if options.get('out') is not None:
        outFileName = options.get('out')
        if outFileName=='':
            for i in range(100):
                outFileName = os.path.splitext(options.get('in'))[0] + 'R%d.fbm'%(i+1)
                if os.path.exists( outFileName ): continue
                break
        elif options.get('restart'):
            if os.path.exists( options.get('out') ): os.remove( options.get('out') )

    # Initialize the dataset:
    myPrint('\nPreparing CIFAR-100 datasets ... ')
    if options.get('useTest'):
        trainDs, testDs = CiFarAutoDSet.makeDatasets('train,test', batchSize=options.get('batchSize'))
        validDs = None
    else:
        trainDs, testDs, validDs = CiFarAutoDSet.makeDatasets('train,test,valid', batchSize=options.get('batchSize'))
    
    model = Model.makeFromFile(options.get('in'),
                               trainDs=trainDs, testDs=testDs, validationDs=validDs,
                               batchSize=options.get('batchSize'),
                               numEpochs=options.get('epochs'),
                               regFactor=options.get('regFactor'),
                               dropOutKeep=1.0-options.get('dropOut'),
                               learningRate=eval(options.get('learningRate')),
                               optimizer=options.get('optimizer'),
                               saveModelFileName=outFileName,   # Save retrained model
                               savePeriod=1,                    # Save model every epoch
                               saveBest=True,                   # Keep a copy of best network during the training.
                               gpus=options.get('gpus'))

    model.printLayersInfo()
    model.printNetConfig()
    model.initSession()

    model.train()
    if options.get('useTest'):
        myPrint('\nRunning inference on %d Test Samples (batchSize:%d, %d towers) ... ' %
                (testDs.numSamples, testDs.batchSize, len(model.towers)))
        model.evaluateDSet(testDs)
    else:
        myPrint('\nRunning inference on %d Validation Samples (batchSize:%d, %d towers) ... ' %
                (validDs.numSamples, validDs.batchSize, len(model.towers)))
        model.evaluateDSet(validDs)

    if options.get('saveResults') is not None:
        from fireball.paramsearch import saveResults
        resultsCsvFileName = options.get('saveResults')
        paramValues = [(n,options.get(n)) for n in ['regFactor', 'dropOut', 'batchSize', 'optimizer',
                                                    'learningRate', 'epochs'] ]
        if outFileName is not None:
            paramValues.insert(0,('out',outFileName))
            
        saveResults(resultsCsvFileName, paramValues, model)

