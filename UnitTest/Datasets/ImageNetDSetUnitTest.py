"""
***********************************************************************************************************************
Filename:         ImageNetDSetUnitTest.py
Copyright (c) 2020 InterDigital AI Research Lab

Description:
This file contains the Unit Test code for the "ImageNetDSet" class.


Version History:
Date Changed            By                      Description
------------            --------------------    ------------------------------------------------
03/02/2020              Shahab Hamidi-Rad       Created the file.
04/06/2020              Shahab Hamidi-Rad       Added test cases for creating and using tuning
                                                datasets.
"""
from fireball.datasets.imagenet import ImageNetDSet
from fireball import myPrint
import time
import os

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

        if batchSamples.shape[1:] != (224,224,3):
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
    for w in [8, 0]:
        for batchSize in [256, 64, 1]:
            if (w == 0) and (batchSize == 1):   continue
            myPrint('\nTesting only Test dataset (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
            testDs = ImageNetDSet.makeDatasets('Test', numWorkers=w)
            assert testDs is not None, "testDs must not be None!"

            print(testDs)
            testBatches(testDs, batchSize)


    vr = 0.05
    w = 8
    batchSize = 256
    myPrint('\nTesting all datasets (batchSize=%d, ValidRatio=%.2f, numWorkers=%d) ...'%(batchSize, vr, w), color='cyan')
    trainDs, testDs, validDs = ImageNetDSet.makeDatasets('Train,Test,Valid%%%d'%(int(vr*100)), numWorkers=w)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    ImageNetDSet.printDsInfo(trainDs, testDs, validDs)
    testBatches(trainDs, batchSize)
    testBatches(validDs, batchSize)

    w = 8
    batchSize = 256
    myPrint('\nTesting all datasets (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
    trainDs, testDs, validDs = ImageNetDSet.makeDatasets(numWorkers=w)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    ImageNetDSet.printDsInfo(trainDs, testDs, validDs)
    testBatches(trainDs, batchSize)
    testBatches(validDs, batchSize)

    w = 8
    tr = .05
    vr = .05
    batchSize = 256
    myPrint('\nTesting Fine-Tuning dataset (batchSize=%d, TuneRatio=%.2f, ValidRatio=%.2f) ...'%(batchSize,tr,vr), color='cyan')
    trainDs, testDs, validDs = ImageNetDSet.makeDatasets('Fine-Tuning%%%d,Test,Valid%%%d'%(int(tr*100), int(vr*100)), batchSize=batchSize, numWorkers=w)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    ImageNetDSet.printDsInfo(trainDs, testDs, validDs)
    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    tr = .1
    vr = .1
    myPrint('\nTesting creating FineTune dataset file (TuneRatio=%.2f, ValidRatio=%.2f) ...'%(tr,vr), color='cyan')

    # Backup existing one:
    myPrint('    Backing up existing Fine-Tune dataset file ... ', False)
    dataPath = '/data/ImageNet/'
    if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
    if os.path.exists(dataPath+"ILSVRC2012Tune224"):
        if os.path.exists(dataPath+"ILSVRC2012Tune224-BkUnitTest")==False:
            os.rename(dataPath+"ILSVRC2012Tune224",dataPath+"ILSVRC2012Tune224-BkUnitTest")
    
    if os.path.exists(dataPath+"TuneDataset.csv"):
        if os.path.exists(dataPath+"TuneDataset-BkUnitTest.csv")==False:
            os.rename(dataPath+"TuneDataset.csv",dataPath+"TuneDataset-BkUnitTest.csv")
    myPrint('Done')

    ImageNetDSet.createFineTuneDataset(dataPath, ratio=tr, copyImages=False)
    trainDs, testDs, validDs = ImageNetDSet.makeDatasets('Tune,Test,Validation%%%d'%(int(vr*100)), batchSize=batchSize)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    ImageNetDSet.printDsInfo(trainDs, testDs, validDs)
    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)

    tr = .05
    vr = .1
    myPrint('\nTesting creating FineTune dataset file (TuneRatio=%.2f, ValidRatio=%.2f, Copying Images) ...'%(tr,vr), color='cyan')

    os.system('rm -rf %sILSVRC2012Tune224'%(dataPath))
    os.system('rm %sTuneDataset.csv'%(dataPath))

    ImageNetDSet.createFineTuneDataset(dataPath, ratio=tr, copyImages=True)
    trainDs, testDs, validDs = ImageNetDSet.makeDatasets('Tune,Test,Valid%%%d'%(int(vr*100)), batchSize=batchSize)
    assert trainDs is not None, "trainDs must not be None!"
    assert testDs is not None, "testDs must not be None!"
    assert validDs is not None, "validDs must not be None!"
    ImageNetDSet.printDsInfo(trainDs, testDs, validDs)
    testBatches(trainDs, batchSize)
    testBatches(validDs, batchSize)

    myPrint('    Restoring original file ... ', False)
    if os.path.exists(dataPath+"ILSVRC2012Tune224-BkUnitTest"):
        os.system('rm -rf %sILSVRC2012Tune224'%(dataPath))
        os.rename(dataPath+"ILSVRC2012Tune224-BkUnitTest",dataPath+"ILSVRC2012Tune224")
    if os.path.exists(dataPath+"TuneDataset-BkUnitTest.csv"):
        os.system('rm %sTuneDataset.csv'%(dataPath))
        os.rename(dataPath+"TuneDataset-BkUnitTest.csv",dataPath+"TuneDataset.csv")
    myPrint('Done')
