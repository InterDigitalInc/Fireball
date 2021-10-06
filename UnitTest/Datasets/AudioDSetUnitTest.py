# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the Unit Test code for the "AudioDSet" class.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 03/02/2020              Shahab Hamidi-Rad       Created the file.
# **********************************************************************************************************************
from fireball.datasets.audio import AudioDSet
from fireball import myPrint
import os,time

# **********************************************************************************************************************
def testBatches(db, batchSize):
    myPrint('    Running the batch iterator for %s dataset ... '%(db.dsName))
    firstSmallBatch = True
    totalSamples = 0
    t0 = time.time()
    classCounts = [0]*db.numClasses
    for b, (batchSamples, batchLabels) in enumerate(db.batches(batchSize)):
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

        if batchSamples.shape[1:] != (40, 500, 1):
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
    for used4Test in ['Eval', 'Test']:
        for batchSize in [64, 256]:
            myPrint('\nTesting all datasets (batchSize=%d, Using "%s" for test dataset) ...'%(batchSize, used4Test),
                    color='cyan')
            trainDs, testDs, validDs = AudioDSet.makeDatasets('Train,%s,Valid'%(used4Test), batchSize=batchSize)
            AudioDSet.printDsInfo(trainDs, testDs, validDs)
            AudioDSet.printStats(trainDs, testDs, validDs)
            
            testBatches(trainDs, batchSize)
            testBatches(testDs, batchSize)
            testBatches(validDs, batchSize)
            

    batchSize = 128
    myPrint('\nTesting only test dataset (batchSize=%d) ...'%(batchSize), color='cyan')
    testDs = AudioDSet.makeDatasets('Test', batchSize=batchSize)
    print(testDs)
    testBatches(testDs, batchSize)

    myPrint('\nTesting only test dataset (batchSize=1) ...', color='cyan')
    testDs = AudioDSet.makeDatasets('Test', batchSize=1)
    testBatches(testDs, 1)

    ratio = 0.1
    batchSize = 128
    myPrint('\nTesting Fine Tuning (batchSize=%d, Ratio=%.2f) ...'%(batchSize, ratio), color='cyan')
    trainDs, testDs, validDs = AudioDSet.makeDatasets('Tune%%%d,Test,Valid'%(int(ratio*100)), batchSize=batchSize)
    AudioDSet.printDsInfo(trainDs, testDs, validDs)
    AudioDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    ratio = 0.1
    batchSize = 128
    myPrint('\nTesting Split function (batchSize=%d, Ratio=%.2f) ...'%(batchSize, ratio), color='cyan')
    trainDs = AudioDSet.makeDatasets('Train', batchSize=batchSize)

    print(trainDs)
    validDs = trainDs.split(ratio=ratio)

    AudioDSet.printDsInfo(trainDs, None, validDs)
    AudioDSet.printStats(trainDs, None, validDs)

    testBatches(trainDs, batchSize)
    testBatches(validDs, batchSize)

    tr = .2
    myPrint('\nTesting creating Fine-Tune dataset file (TuneRatio=%.2f) ...'%(tr), color='cyan')

    # Backup existing one:
    dataPath = '/data/AudioDb2/'
    if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
    if os.path.exists(dataPath+"TuneDb.npz"):
        if os.path.exists(dataPath+"TuneDb.bak")==False:
            myPrint('    Backing up existing Fine-Tune dataset file ... ', False)
            os.rename(dataPath+"TuneDb.npz",dataPath+"TuneDb.bak")
            myPrint('Done')

    AudioDSet.createFineTuneDataset(ratio=tr)
    trainDs, testDs, validDs = AudioDSet.makeDatasets('Tune,test,valid', batchSize=batchSize)
    AudioDSet.printDsInfo(trainDs, testDs, validDs)
    AudioDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    if os.path.exists(dataPath+"TuneDb.bak"):
        myPrint('    Restoring original file ... ', False)
        os.system('rm %sTuneDb.npz'%(dataPath))
        os.rename(dataPath+"TuneDb.bak",dataPath+"TuneDb.npz")
        myPrint('Done')
