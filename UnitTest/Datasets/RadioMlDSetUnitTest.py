"""
************************************************************************************************************************
Filename:         RadioMlDSetUnitTest.py

Description:
This file contains the Unit Test code for the "RadioMlDSet" class.

Copyright (c) 2020 InterDigital AI Research Lab

Version History:
Date Changed            By                      Description
------------            --------------------    ------------------------------------------------
02/28/2020              Shahab Hamidi-Rad       Created the file.
"""
from fireball.datasets.radioml import RadioMlDSet
from fireball import myPrint
import os, time

# **********************************************************************************************************************
def testBatches(ds, batchSize):
    myPrint('    Running the batch iterator for %s dataset (batchSize=%d)... '%(ds.dsName, batchSize))
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

        
        if batchSamples.shape[1:] != (128 if ds.version==2016 else 1024,1, 2):
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
    RadioMlDSet.createNpzFiles()
    
    for version in [2016, 2018]:
        RadioMlDSet.configure(version=version)
        for batchSize in [64, 256]:
            for snrVals in [ -10, [12], [0, 10, 18] ]:
                myPrint('\nTesting all datasets (SNR=%s, batchSize=%d, version=%d) ...'%(str(snrVals), batchSize, version), color='cyan')
                trainDs, testDs, validDs = RadioMlDSet.makeDatasets(snrValues=snrVals, version=version)
                assert trainDs is not None, "trainDs must not be None!"
                assert testDs is not None, "testDs must not be None!"
                assert validDs is not None, "validDs must not be None!"

                RadioMlDSet.printDsInfo(trainDs, testDs, validDs)
                RadioMlDSet.printStats(trainDs, testDs, validDs)

                testBatches(trainDs, batchSize)
                testBatches(testDs, batchSize)
                testBatches(validDs, batchSize)

                myPrint('\nTesting only Test dataset (SNR=%s, batchSize=%d, version=%d) ...'%(str(snrVals), batchSize, version), color='cyan')
                testDs = RadioMlDSet.makeDatasets('test', snrValues=snrVals, version=version)
                assert testDs is not None, "testDs must not be None!"
                print(testDs)
                testBatches(testDs, batchSize)

        batchSize = 128
        myPrint('\nTesting Train/Test datasets (SNR=0, batchSize=%d, version=%d) ...'%(batchSize, version), color='cyan')
        trainDs, testDs = RadioMlDSet.makeDatasets('Train,Test', batchSize=batchSize, snrValues=0, version=version)
        assert trainDs is not None, "trainDs must not be None!"
        assert testDs is not None, "testDs must not be None!"
        RadioMlDSet.printDsInfo(trainDs, testDs)
        RadioMlDSet.printStats(trainDs, testDs)
        testBatches(trainDs, batchSize)
        testBatches(testDs, batchSize)

        vr = .05
        RadioMlDSet.configure(validRatio=vr, version=version)
        batchSize = 128
        snrVals = [0, 10, 18]
        myPrint('\nTesting all datasets (SNR=%s, batchSize=%d, Validation Ratio=%.2f, Repeatable, version=%d) ...'%(str(snrVals), batchSize, vr, version), color='cyan')
        trainDs, testDs, validDs = RadioMlDSet.makeDatasets('Train,Test,Valid', batchSize=batchSize,
                                                            snrValues=snrVals, version=version)
        assert trainDs is not None, "trainDs must not be None!"
        assert testDs is not None, "testDs must not be None!"
        assert validDs is not None, "validDs must not be None!"
        RadioMlDSet.printDsInfo(trainDs, testDs, validDs)
        RadioMlDSet.printStats(trainDs, testDs, validDs)
        testBatches(trainDs, batchSize)
        testBatches(testDs, batchSize)
        testBatches(validDs, batchSize)

        batchSize = 1
        snrVals = [0, 10, 18]
        myPrint('\nTesting Test dataset only (SNR=%s, batchSize=%d, version=%d) ...'%(str(snrVals), batchSize, version), color='cyan')
        testDs = RadioMlDSet.makeDatasets('Test', batchSize=batchSize, snrValues=snrVals, version=version)
        assert testDs is not None, "testDs must not be None!"
        print(testDs)
        testBatches(testDs, 1)

        batchSize = 128
        myPrint('\nTesting Train/Test datasets with all SNRs (batchSize=%d, version=%d) ...'%(batchSize, version), color='cyan')
        trainDs, testDs = RadioMlDSet.makeDatasets('Train,Test', batchSize=batchSize, version=version)
        assert trainDs is not None, "trainDs must not be None!"
        assert testDs is not None, "testDs must not be None!"
        RadioMlDSet.printDsInfo(trainDs, testDs)
        RadioMlDSet.printStats(trainDs, testDs)
        testBatches(trainDs, batchSize)
        testBatches(testDs, batchSize)

        tr = .05
        vr = .05
        RadioMlDSet.configure(tuneRatio=tr, validRatio=vr, version=version)
        myPrint('\nTesting Fine-Tuning datasets (SNR=%s, batchSize=%d, TuneRatio=%.2f, ValidRatio=%.2f, Repeatable, version=%d) ...'%(str(snrVals),batchSize,tr,vr, version), color='cyan')
        trainDs, testDs, validDs = RadioMlDSet.makeDatasets('Tune,test,valid', batchSize=batchSize,
                                                            snrValues=snrVals, version=version)
        assert trainDs is not None, "trainDs must not be None!"
        assert testDs is not None, "testDs must not be None!"
        assert validDs is not None, "validDs must not be None!"

        RadioMlDSet.printDsInfo(trainDs, testDs, validDs)
        RadioMlDSet.printStats(trainDs, testDs, validDs)

        testBatches(trainDs, batchSize)
        testBatches(testDs, batchSize)
        testBatches(validDs, batchSize)
