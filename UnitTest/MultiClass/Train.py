import numpy as np
import os

from fireball import Model, Block, DashOptions, myPrint
from fireball.datasets.mnist import MnistDSet

# ************************************************************************************************
options = DashOptions(
    [
        ('help',     'Show this help',               ['bool']),
        ('h',        None,                           ['bool']),  # No Description -> hidden option(will not appear in the usage)

        ('out',             'Name of output model (default: Models/LeNet5.tfn)',                ['string','noValDef:']),

        ('batchSize',      'The batch size used in each iteration of training.(Default: 128)',  ['int', 'min:32', 'default:128']),
        ('epochs',         'Number of Epochs for NN algorithm.(Default: 10)',                   ['int', 'min:1', 'default:10']),
        ('regFactor',      'Regularization Factor for NN algorithm.(Default: 0)',               ['float', 'min:0.0', 'default:0']),
        ('dropOut',        'Drop Out Probability for NN algorithm.(Default: 0)',                ['float', 'min:0.0', 'max:0.9', 'default:0']),
        ('learningRate',   'Can be one of the following:(Default: 0.1,0.0001)\n'+
                           'lrValue: Fixed learning rate during the training.\n'+
                           '"(start, end)": learning rate starts at "start" ending in "end".\n'+
                           '"[(1,lr1),(N2,lr2),...,(nN,lrN)]" Piecewise learning rate.',        ['string', 'default:(0.1,0.0001)']),
        ('optimizer',      '"GradientDescent", "Adam", or "Momentum".(Default: Momentum)',      ['string', 'default:Momentum']),

        ('layers',         'The choice of layers to use for the model.',                        ['int', 'min:1', 'max:4', 'default:1']),

        ('gpus',           'The GPUs used for retraining.',                                     ['string', 'default:0']),
        ('restart',        'Delete intermediate files and restart the process from scratch.',   ['bool']),
        
        ('paramSearch',    'Do a grid search on the hyper parameters of the model.',            ['bool']),
        ('numWorkers',     'The number of worker processes for parameter search.(Default: 1)',  ['int', 'default:1']),
        ('log',            'log the parameter search progress to a file.',                      ['bool']),
        ('saveResults',    'Save results to specified csv file.',                               ['string']),
        ('useTest',        'Use test set for evaluation. Otherwise use validation set.',        ['bool']),
     ])

# ************************************************************************************************
def getLayersInfo(choice):
    if choice==1:
        # Lenet5 with batchnorm:
        layersInfo = 'IMG_S28_D1,CONV_K5_O6_Ps:ReLU:MP_K2,CONV_K5_O16_Pv,BN:ReLU:MP_K2,FC_O120:ReLU,FC_O84:ReLU,FC_O10:None,CLASS_C10'
        return layersInfo, []

    if choice==2:
        # Lenet5 with low-rank layers:
        layersInfo = 'IMG_S28_D1,CONV_K5_O6_Ps:ReLU:MP_K2,CONV_K5_O16_R4_Pv:ReLU:MP_K2,FC_O120_R8:ReLU,FC_O84:ReLU,FC_O10:None,CLASS_C10'
        return layersInfo, []

    if choice==3:
        # Testing mobileNet Blocks
        layersInfo = 'IMG_S28_D1,CONV_K5_O6_Ps:ReLU:MP_K2,CONV_K5_O16_Pv:ReLU:MP_K2,MN1S_X64_O16:ReLU:MP_K2,FC_O120:ReLU,FC_O84:ReLU,FC_O10:None,CLASS_C10'
        blocks = [ Block('MN1|x_expansion_i,o_outDept_i|' +     # MobileNet Block with Stride 1 No shortcut
                         'add|' +
                         'CONV_K1_O%x_Ps_B0,BN:ReLU:CLP_H6,DWCN_K3_S1_Ps_B0,BN:ReLU:CLP_H6,CONV_K1_O%o_Ps_B0,BN'),
    
                   Block('MN1S|x_expansion_i,o_outDept_i|' +    # MobileNet Block with Stride 1 With shortcut
                         'add|' +
                         'CONV_K1_O%x_Ps_B0,BN:ReLU:CLP_H6,DWCN_K3_Ps_B0,BN:ReLU:CLP_H6,CONV_K1_O%o_Ps_B0,BN;ID'),
    
                   Block('MN2|x_expansion_i,o_outDept_i|' +     # MobileNet Block with Stride 2 No shortcut
                         'add|' +
                         'CONV_K1_O%x_Ps_B0,BN:ReLU:CLP_H6,DWCN_K3_S2_P0x1x0x1_B0,BN:ReLU:CLP_H6,CONV_K1_O%o_Ps_B0,BN') ]
        return layersInfo, blocks

    if choice==4:
        # Testing ResNet Blocks:
        layersInfo = 'IMG_S28_D1,CONV_K5_O6_Ps:ReLU:MP_K2,CONV_K5_O16_Pv:ReLU:MP_K2,RES2_K3_O16/16/32_S1:ReLU,RES1_K3_O16/16/32:ReLU,FC_O120:ReLU,FC_O84:ReLU,FC_O10:None,CLASS_C10'
        blocks = [ # This is id_block in Keras Model
                   Block('RES1|k_kernel_ixi,o_outSizes_i*3,s_stride_ixi_1|' +
                         'add|' +
                         'CONV_K1_S%s_O%o0_Pv,BN:ReLU,CONV_K%k_S1_O%o1_Ps,BN:ReLU,CONV_K1_S1_O%o2,BN;ID'),
                   # This is conv_block in Keras Model
                   Block('RES2|k_kernel_ixi,o_outSizes_i*3,s_stride_ixi_1|' +
                         'add|' +
                         'CONV_K1_S%s_O%o0_Pv,BN:ReLU,CONV_K%k_S1_O%o1_Ps,BN:ReLU,CONV_K1_S1_O%o2,BN;'+
                         'CONV_K1_S%s_O%o2_Pv,BN') ]
        return layersInfo, blocks

    assert False, "Invalid Layers Choice (%d)!"%(choice)

# ************************************************************************************************
if __name__ == '__main__':
    if options.run() == False:
        exit(1)

    if options.get('paramSearch'):
        from fireball.paramsearch import paramSearch
        # Param Names here must match the ones in the command line options defined above.
        parameterChoices = [
                            ('regFactor',    [0.0, 0.001]),
                            ('dropOut',      [0.1]),
                            ('batchSize',    [128]),
                            ('learningRate', ["(0.1,0.0001)", "[(0,.1),(500,.01),(1500,.001)]"]),
                            ('optimizer',    ['Momentum']),
                            ('epochs',       [5]),
                           ]

        additionalParams = {}
        
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
                outFileName = 'Models/Test%d.fbm'%(i+1)
                if os.path.exists( outFileName ): continue
                break
        elif options.get('restart'):
            if os.path.exists( options.get('out') ): os.remove( options.get('out') )

    # Initialize the dataset:
    myPrint('\nPreparing MNIST datasets ... ')
    if options.get('useTest'):
        trainDs, testDs = MnistDSet.makeDatasets('train,test', batchSize=options.get('batchSize'))
        validDs = None
    else:
        trainDs, testDs, validDs = MnistDSet.makeDatasets('train,test,valid', batchSize=options.get('batchSize'))
    MnistDSet.printStats(trainDs, testDs, validDs)
    
    layersInfo, blocks = getLayersInfo( options.get('layers') )
    model = Model(name='ClassTest', layersInfo=layersInfo,
                  blocks=blocks,
                  trainDs=trainDs, testDs=testDs, validationDs=validDs,
                  batchSize=options.get('batchSize'),
                  numEpochs=options.get('epochs'),
                  regFactor=options.get('regFactor'),
                  dropOutKeep=1.0-options.get('dropOut'),
                  learningRate=eval(options.get('learningRate')),
                  optimizer=options.get('optimizer'),
                  saveModelFileName=outFileName,           # Save trained model
                  savePeriod=1,                            # Save model every epoch
                  saveBest=True,                           # Keep a copy of best network during the training.
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
        model.evaluateDSet(validDs, topK=3)

    if options.get('saveResults') is not None:
        from fireball.paramsearch import saveResults
        resultsCsvFileName = options.get('saveResults')
        paramValues = [(n,options.get(n)) for n in ['regFactor', 'dropOut', 'batchSize', 'optimizer',
                                                    'learningRate', 'epochs'] ]
        if outFileName is not None:
            paramValues.insert(0,('out',outFileName))
            
        saveResults(resultsCsvFileName, paramValues, model)

