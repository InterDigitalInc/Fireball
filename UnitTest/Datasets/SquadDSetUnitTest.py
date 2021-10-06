# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the Unit Test code for the "SquadDSet" class.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 07/24/2020    Shahab Hamidi-Rad       Created the file.
# 07/29/2020    Shahab Hamidi-Rad       Completed all test cases.
# **********************************************************************************************************************
from fireball.datasets.squad import SquadDSet
from fireball import myPrint
import os, time

# **********************************************************************************************************************
def testBatches(ds, batchSize, numWorkers=0):
    if numWorkers==0:   myPrint('    Running the batch iterator for %s dataset ... '%(ds.dsName))
    else:               myPrint('    Running the batch iterator for %s dataset (%d workers)... '%(ds.dsName,numWorkers))
    firstSmallBatch = True
    totalSamples = 0
    t0 = time.time()
    numImpossible = 0
    for b, (batchSamples, batchLabels) in enumerate(ds.batches(batchSize,numWorkers)):
        myPrint('    Testing batch %d - (Total Samples so far: %d) ... \r'%(b, totalSamples), False)
        batchSampleIdxs, batchTokenIds, batchTokenTypes = batchSamples
        totalSamples += len(batchTokenIds)

        if len(batchTokenIds) != batchSize:
            if firstSmallBatch:
                firstSmallBatch = False
            else:
                myPrint( '\nError: Only the last batch can be smaller than the batch size!', color='red')
                exit(1)
        
        if len(batchTokenIds) != len(batchTokenTypes):
            myPrint( '\nError: The number of samples in "batchTokenIds" and "batchTokenTypes" should match!', color='red')
            exit(1)
        
        if type(batchLabels) == tuple:
            batchLabels0, batchLabels1 = batchLabels
            if len(batchTokenIds) != len(batchLabels0):
                myPrint( '\nError: The number of samples(%d) should match the number of labels(%d)!'%(len(batchTokenIds), len(batchLabels0)), color='red')
                exit(1)

            if len(batchLabels0) != len(batchLabels1):
                myPrint( '\nError: The number of labels in first(%d) and second(%d) components should match!'(len(batchLabels0), len(batchLabels1)), color='red')
                exit(1)
                
        for i,(seq,types) in enumerate(zip(batchTokenIds, batchTokenTypes)):
            if len(seq) != ds.maxSeqLen:
                myPrint( '\nError: The length of sequences in "batchTokenIds" should match "maxSeqLen"!', color='red')
                exit(1)
            if len(types) != ds.maxSeqLen:
                myPrint( '\nError: The length of sequences in "batchTokenIds" should match "maxSeqLen"!', color='red')
                exit(1)

            try:        questionLen = seq.index( ds.tokenizer.sep['id'] )-1
            except:     questionLen = 0
            try:        contextLen = seq[2+questionLen:].index( ds.tokenizer.sep['id'] )
            except:     contextLen = -1
            if (questionLen<=0) or (contextLen<0) or (seq[0] != ds.tokenizer.cls['id']):
                myPrint( '\nError: Sequence should contain "[CLS] + question + [SEP] + context + [SEP] + Padding"!', color='red')
                exit(1)

            if questionLen<4:
                question = ds.tokenizer.decode(seq[1:1+questionLen])
                myPrint( '\nError: Invalid question length (%d-"%s")!'%(questionLen,question), color='red')
                exit(1)

            if contextLen<10:
                context = ds.tokenizer.decode(seq[1:1+contextLen])
                myPrint( '\nError: Invalid context length (%d-"%s")!'%(contextLen, context), color='red')
                exit(1)

            try:    questionTypeLen = types.index(1) - 2
            except: questionTypeLen = -1
            if questionTypeLen != questionLen:
                myPrint( '\nError: Question length does not match in sequence '
                         'tokens(%d) and token types(%d)"!'%(questionLen,questionTypeLen), color='red')
                exit(1)

            try:    contextTypeLen = types[questionTypeLen+2:].index(0)-1
            except: contextTypeLen = len(types)-questionTypeLen-3
            if contextTypeLen != contextLen:
                myPrint( '\nError: Context length does not match in sequence '
                         'tokens(%d) and token types(%d)"!'%(contextLen,contextTypeLen), color='red')
                exit(1)

            if ds.isTraining:
                startPos, endPos = batchLabels0[i], batchLabels1[i]
                if ds.version<2:
                    if startPos==0 and endPos==0:
                        myPrint( '\nError: Impossible questions are not supported for SQuAD version 1!', color='red')
                        exit(1)
                
                if startPos==0 and endPos==0:
                    numImpossible += 1
                    continue
                    
                if startPos>endPos:
                    myPrint( '\nError: StartPos(%d) of answer should not be larger than its endPos(%d)!'%(startPos,endPos), color='red')
                    exit(1)

                if startPos<(questionLen+2):
                    myPrint( '\nError: StartPos(%d) should be larger than question length(%d) plus 2!'%(startPos,contextLen), color='red')
                    exit(1)

                noPadLen = questionLen+contextLen+3
                if endPos>noPadLen:
                    myPrint( '\nError: EndPos(%d) should not be larger than noPadLength(%d) plus 2!'%(endPos,noPadLen), color='red')
                    exit(1)
            else:
                answerTexts = batchLabels[i]
                impossible = True
                for answerText in answerTexts:
                    if answerText != "":
                        impossible=False
                        break

                if ds.version<2 and impossible:
                    myPrint( '\nError: Impossible questions are not supported for SQuAD version 1!', color='red')
                    exit(1)

                if impossible:
                    numImpossible += 1
                    continue
                    
    if ds.stats['NumImpossible'] != numImpossible:
        myPrint( '\nError: Number of impossible questions(%d) does not match Dataset stats(%d)!'%(numImpossible,ds.stats['NumImpossible']), color='red')
        exit(1)

    myPrint('    Success! (%d batches, %d Samples, %.2f Sec.)                 '%(b+1, totalSamples, time.time()-t0),
            color='green')

# **********************************************************************************************************************
def testContextTokSpans(ds):
    for i,context in enumerate(ds.contexts):
        contextSpans = ds.contextsTokSpans[i]
        text = ""
        for start,end in contextSpans:
            text += context[start:end]
        if text != context:
            myPrint( '\nError: Context span test failed:\nContext:\n%s\nReconstructed from Spans:\n%s\n'%(context,text), color='red')
            exit(1)
            
    myPrint('    Success! (Tested %d contexts)'%(len(ds.contexts)), color='green')

# **********************************************************************************************************************
if __name__ == '__main__':
    for ver in [1,2]:
        for batchSize in [16,8,1]:
            if batchSize==16:
                myPrint('\nLoading SQuAD dataset (Version %d) ...'%(ver), color='cyan')
                trainDs, testDs = SquadDSet.makeDatasets(batchSize=batchSize, version=ver )
                assert trainDs is not None, "trainDs must not be None!"
                assert testDs is not None, "testDs must not be None!"

                SquadDSet.printDsInfo(trainDs, testDs)
                SquadDSet.printStats(trainDs, testDs)
                
                myPrint('\nTesting Context Spans on Train dataset (SQuAD version %d) ...'%(ver), color='cyan')
                testContextTokSpans(trainDs)
                
                myPrint('\nTesting Context Spans on Test dataset (SQuAD version %d) ...'%(ver), color='cyan')
                testContextTokSpans(testDs)

            myPrint('\nTesting dataset (SQuAD version %d, batchSize=%d) ...'%(ver,batchSize), color='cyan')
            testBatches(trainDs, batchSize, numWorkers=(4 if batchSize==16 else 0))
            testBatches(testDs, batchSize)

    myPrint('\nTesting evaluation ...', color='cyan')
    testDs = SquadDSet.makeDatasets("Test", batchSize=8, version=2 )
    assert testDs is not None, "testDs must not be None!"
    SquadDSet.printDsInfo(None, testDs)
    SquadDSet.printStats(None, testDs)
    
    results = testDs.evaluate("UnitTestPreds.json")
    expectedResults = {
        "exact":        64.81091552261434,
        "f1":           67.60971132981282,
        "numQuestions": 11873,
        "hasAnsExact":  59.159919028340084,
        "hasAnsF1":     64.76553687902589,
        "numHasAns":    5928,
        "noAnsExact":   70.4457527333894,
        "noAnsF1":      70.4457527333894,
        "numNoAns":     5945
    }
    for key,val in expectedResults.items():
        if abs(val-results[key]) > 0.000001:
            myPrint( '\nError: The value of "%s" in the results (%s) does not match the expected'
                     'value (%s)!'%(key, str(results[key]), str(val)), color='red')
            exit(1)
    myPrint('    Success! (Exact Match: %f, F1: %f)'%(results["exact"], results["f1"]), color='green')

