"""
************************************************************************************************************************
Filename:         MnistDSetUnitTest.py

Description:
This file contains the Unit Test code for the "MnistDSet" class.

Copyright (c) 2020 InterDigital AI Research Lab

Version History:
Date Changed            By                      Description
------------            --------------------    ------------------------------------------------
03/02/2020              Shahab Hamidi-Rad       Created the file.
"""
from fireball.datasets.mnist import MnistDSet
from fireball import myPrint
import os, time

# **********************************************************************************************************************
def testBatches(ds, batchSize):
    myPrint('    Running the batch iterator for %s dataset ... '%(ds.dsName))

    firstSmallBatch = True
    totalSamples = 0
    t0 = time.time()
    classCounts = [0]*ds.numClasses
    for b, (batchSamples, batchLabels) in enumerate(ds.batches(batchSize)):
        myPrint('    Testing batch %d - (Total Samples so far: %d) ... \r'%(b, totalSamples), False)
        if batchSamples.shape[0] != batchSize:
            if firstSmallBatch:
                firstSmallBatch = False
            else:
                myPrint( '\nError: Only the last batch can be smaller than the batch size!', color='red')
                exit(0)

        if batchSamples.shape[0] != batchLabels.shape[0]:
            myPrint( '\nError: The number of samples and labels in a batch should match!', color='red')
            exit(0)

        if batchSamples.shape[1:] != (28, 28, 1):
            myPrint( '\nError: Invalid Batch Samples Shape: ' + str(batchSamples.shape), color='red')
            exit(0)
        totalSamples += batchSamples.shape[0]
        for l in batchLabels:   classCounts[l] += 1

    myPrint('    Success! (%d batches, %d Samples, %.2f Sec., Class Counts(min/max):%d/%d)                 '%(b+1,
                                                                                    totalSamples, time.time()-t0,
                                                                                    min(classCounts), max(classCounts)),
            color='green')

# **********************************************************************************************************************
if __name__ == '__main__':
    for batchSize in [64, 512]:
        myPrint('\nTesting all datasets (batchSize=%d) ...'%(batchSize), color='cyan')
        trainDs, testDs, validDs = MnistDSet.makeDatasets(batchSize=batchSize)
        assert trainDs is not None, "trainDs must not be None!"
        assert testDs is not None, "testDs must not be None!"
        assert validDs is not None, "validDs must not be None!"
        MnistDSet.printDsInfo(trainDs, testDs, validDs)
        MnistDSet.printStats(trainDs, testDs, validDs)

        testBatches(trainDs, batchSize)
        testBatches(testDs, batchSize)
        testBatches(validDs, batchSize)
            
    vr = .05
    batchSize = 128
    myPrint('\nTesting all datasets (batchSize=%d, Validation Ratio=%.2f) ...'%(batchSize, vr), color='cyan')
    trainDs, testDs, validDs = MnistDSet.makeDatasets('Train,Test,Valid%%%d'%(int(vr*100)), batchSize=batchSize)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    MnistDSet.printDsInfo(trainDs, testDs, validDs)
    MnistDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    myPrint('\nTesting Test dataset Only (batchSize=%d) ...'%(batchSize), color='cyan')
    testDs = MnistDSet.makeDatasets('Test', batchSize=batchSize)
    assert testDs is not None, "testDs must not be None!"
    print(testDs)
    testBatches(testDs, batchSize)

    myPrint('\nTesting Test dataset Only (batchSize=1) ...', color='cyan')
    testDs = MnistDSet.makeDatasets('Test', batchSize=1)
    assert testDs is not None, "testDs must not be None!"
    testBatches(testDs, 1)

    tr = .05
    vr = .05
    myPrint('\nTesting all dataset (FineTuning, batchSize=%d, TuneRatio=%.2f, ValidRatio=%.2f) ...'%(batchSize,tr,vr), color='cyan')
    trainDs, testDs, validDs = MnistDSet.makeDatasets('Tune%%%d,test,valid%%%d'%(int(tr*100), int(vr*100)), batchSize=batchSize)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"

    MnistDSet.printDsInfo(trainDs, testDs, validDs)
    MnistDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)
    
    tr = .2
    vr = .1
    myPrint('\nTesting creating FineTune dataset file (TuneRatio=%.2f, ValidRatio=%.2f) ...'%(tr,vr), color='cyan')

    # Backup existing one:
    dataPath = '/data/mnist/'
    if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
    if os.path.exists(dataPath+"TuneDb.npz"):
        if os.path.exists(dataPath+"TuneDb.bak")==False:
            myPrint('    Backing up existing Fine-Tune dataset file ... ', False)
            os.rename(dataPath+"TuneDb.npz",dataPath+"TuneDb.bak")
            myPrint('Done')

    MnistDSet.createFineTuneDataset(ratio=tr)
    trainDs, testDs, validDs = MnistDSet.makeDatasets('Tune,test,valid%%%d'%(int(vr*100)), batchSize=batchSize)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    MnistDSet.printDsInfo(trainDs, testDs, validDs)
    MnistDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    if os.path.exists(dataPath+"TuneDb.bak"):
        myPrint('    Restoring original file ... ', False)
        os.system('rm %sTuneDb.npz'%(dataPath))
        os.rename(dataPath+"TuneDb.bak",dataPath+"TuneDb.npz")
        myPrint('Done')

