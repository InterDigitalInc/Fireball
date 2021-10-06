# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the Unit Test code for the "GlueDSet" class.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 08/21/2020    Shahab Hamidi-Rad       Created the file.
# **********************************************************************************************************************
from fireball.datasets.glue import GlueDSet
from fireball import myPrint
import os, time

# **********************************************************************************************************************
def testBatches(ds, batchSize, numWorkers=0):
    if numWorkers==0:   myPrint('    Running the batch iterator for %s dataset ... '%(ds.dsName))
    else:               myPrint('    Running the batch iterator for %s dataset (%d workers)... '%(ds.dsName,numWorkers))
    firstSmallBatch = True
    totalSamples = 0
    t0 = time.time()
    for b, (batchSamples, batchLabels) in enumerate(ds.batches(batchSize,numWorkers)):
        myPrint('    Testing batch %d - (Total Samples so far: %d) ... \r'%(b, totalSamples), False)
        totalSamples += len(batchSamples[0])

        if len(batchSamples[0]) != batchSize:
            if firstSmallBatch:
                firstSmallBatch = False
            else:
                myPrint( '\nError: Only the last batch can be smaller than the batch size!', color='red')
                exit(1)

        for batchComponent in batchSamples[1:]:
            if len(batchComponent) != len(batchSamples[0]):
                myPrint( '\nError: The number of samples in different components must match!', color='red')
                exit(1)
        
        for i,(tokIds, tokTypes) in enumerate(zip(batchSamples[1], batchSamples[2])):
            if len(tokIds) != len(tokTypes):
                myPrint( '\nError: The length of tokenIds (%d) and tokenTypes(%d) do not match!'%(len(tokIds), len(tokTypes)),
                         color='red')
                exit(1)

            if len(tokIds) != (ds.maxSeqLen):
                myPrint( '\nError: The length of tokenIds (%d) should match "maxSeqLen" (%d)!'%(len(tokIds), ds.maxSeqLen),
                        color='red')
                exit(1)

    myPrint('    Success! (%d batches, %d Samples, %.2f Sec.)                 '%(b+1, totalSamples, time.time()-t0),
            color='green')

# **********************************************************************************************************************
if __name__ == '__main__':
    for task in ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI-M', 'MNLI-MM', 'QNLI', 'RTE', 'WNLI', 'SNLI']:
        for batchSize in [16,8,1]:
            if batchSize==16:
                myPrint('\nLoading GLUE dataset (Task %s) ...'%(task), color='cyan')
                trainDs, devDs, testDs = GlueDSet.makeDatasets(task, 'Train,Dev,Test', batchSize=batchSize )
                GlueDSet.printDsInfo(trainDs, devDs, testDs)
                GlueDSet.printStats(trainDs, devDs, testDs)
                                
            myPrint('\nTesting GLUE dataset (Task %s, batchSize=%d) ...'%(task,batchSize), color='cyan')
            testBatches(trainDs, batchSize, numWorkers=(4 if batchSize==16 else 0))
            testBatches(devDs, batchSize)
            testBatches(testDs, batchSize)

