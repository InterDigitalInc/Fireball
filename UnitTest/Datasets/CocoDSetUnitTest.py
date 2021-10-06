"""
***********************************************************************************************************************
Filename:         CocoDSetUnitTest.py
Copyright (c) 2020 InterDigital AI Research Lab

Description:
This file contains the Unit Test code for the "CocoDSet" class.

Version History:
Date Changed            By                      Description
------------            --------------------    ------------------------------------------------
03/02/2020              Shahab Hamidi-Rad       Created the file.
"""
from fireball.datasets.coco import CocoDSet
from fireball import myPrint
import time
import os
import numpy as np

# **********************************************************************************************************************
def testBatches(db, batchSize):
    myPrint('    Running the batch iterator for %s dataset ... '%(db.dsName))
    firstSmallBatch = True
    totalSamples = 0
    t0 = time.time()
    for b, (batchSamples, batchLabels) in enumerate(db.batches(batchSize)):
        myPrint('      Testing batch %d - (Total Samples so far: %d) ... \r'%(b, totalSamples), False)
        if batchSamples.shape[0] != batchSize:
            if firstSmallBatch:
                firstSmallBatch = False
            else:
                myPrint( '\n    Error: Only the last batch can be smaller than the batch size!\n', color='red')
                exit(0)

        if batchSamples.shape[1:] != db.sampleShape:
            myPrint( '\n    Error: Invalid Batch Samples Shape: %s\n'%str(batchSamples.shape), color='red')
            exit(0)

        if db.anchorBoxes is not None:
            numAncherBoxes = db.anchorBoxes.shape[0]
            gtBatchLabels, gtBatchBoxAdj, gtBatchMask = batchLabels
            if gtBatchLabels.shape != (batchSamples.shape[0], numAncherBoxes):
                myPrint( '\n    Error: Invalid ground truth labels shape: %s\n'%str(gtBatchLabels.shape), color='red')
                exit(0)
            if gtBatchBoxAdj.shape != (batchSamples.shape[0], numAncherBoxes, 4):
                myPrint( '\n    Error: Invalid ground truth boxes shape: %s\n'%str(gtBatchBoxAdj.shape), color='red')
                exit(0)
            if gtBatchMask.shape != (batchSamples.shape[0], numAncherBoxes):
                myPrint( '\n    Error: Invalid ground truth masks shape: %s\n'%str(gtBatchMask.shape), color='red')
                exit(0)

        elif batchSamples.shape[0] != len(batchLabels):
            myPrint( '\n    Error: The number of samples and labels in a batch should match!\n', color='red')
            exit(0)
            
        totalSamples += batchSamples.shape[0]

    myPrint('    Success! (%d batches, %d Samples, %.2f Sec.)                 '%(b+1,
                                                                                 totalSamples, time.time()-t0),
            color='green')

# **********************************************************************************************************************
def testAnchorBoxes():
    # Visual Test of Anchor/Ground-Truth boxes
    myPrint('\nTesting Anchor/Ground-Truth boxes (Interactive) ...', color='cyan')
    if os.path.exists("AnchorBoxes.npz")==False:
        myPrint('    Ground Truth test data not available (AnchorBoxes.npz). Skipping this test.', color='yellow')
        return
        
    if os.path.exists("cat_dog.jpg")==False:
        myPrint('    Ground Truth test image not available (cat_dog.jpg). Skipping this test.', color='yellow')
        return

    rootDic = np.load("AnchorBoxes.npz", encoding='latin1')
    anchorBoxes = rootDic['AnchorBoxes']
    testDs = CocoDSet.makeDatasets('Test')
    testDs.setAcnchorBoxes(anchorBoxes)
    
    import cv2
    img = np.float32(cv2.imread("cat_dog.jpg")) # BGR format
    dogBox = np.float32([54, 176, 756, 588])
    catBox = np.float32([674, 260, 938, 569])
    boxes = np.float32([ dogBox, catBox ])
    boxes = CocoDSet.p1P2ToP1Size(boxes)
    scores = [.6, .8]
    labels = np.int32([17, 16])

    try:
        # Tesing original sample:
        myPrint('    Close the image windows to go to the next step.')
        CocoDSet.showImageAndBoxes(img, boxes, labels, "Original Sample")
    except:
        myPrint('    No Display Capability! Skipping this test.', color='yellow')
        return

    anchorCenters, anchorSizes = anchorBoxes[:,:2], anchorBoxes[:,2:]

    scaledImg, scaledBoxes, imgSize = CocoDSet.scaleImageAndBoxes(img, boxes, 512, False)

    gtLabels, gtBoxAdj, gtMask, gtIous = testDs.getGroundTruth(labels, scaledBoxes)
    matchingBoxes = []
    matchingLabels = []
    matchingScores = []
    for i in range(len(gtLabels)):
        if gtLabels[i] != 0:
            myPrint("    %d'th gtBox: Label:%s, Box Adjustments:%s, Mask:%d, IOU:%%%.2f"%(i, CocoDSet.classNames[ gtLabels[i] ],
                                                                                          str(gtBoxAdj[i]), gtMask[i],
                                                                                          gtIous[i]*100.0))
            doAdjust = False
            if doAdjust:
                # Apply the adjustments to the anchorBoxes to get the actual boxes. These resulting boxes
                # should match the original boxes in the image exactly.
                centerVar, sizeVar = 0.1, 0.2   # Variance values
                centerAdj, sizeAdj = gtBoxAdj[i][:2], gtBoxAdj[i][2:]
                newCenter = centerAdj * centerVar * anchorSizes[i] + anchorCenters[i]
                newSize = np.exp( sizeAdj * sizeVar ) * anchorSizes[i]
            else:
                # Show the actual anchor boxes without adjustments. This should have and IOU of more than
                # %50 (visually verifiable)
                newCenter, newSize = anchorCenters[i], anchorSizes[i]
            
            matchingBox = np.concatenate((newCenter-newSize/2.0, newSize))
            matchingBoxes += [ matchingBox ]
            matchingLabels += [ gtLabels[i] ]
            matchingScores += [ gtIous[i] ]
            
            showOneByOne = True
            if showOneByOne:
                CocoDSet.showInferResults(img, [np.float32(matchingBox)], [ gtLabels[i] ],
                                          [ gtIous[i] ], False, "Showing Ground Truth box %d"%(i))
                
        elif gtMask[i] == 1:
            myPrint("    Error: %d'th gtBox: Label:%s, Mask:%d"%(i, CocoDSet.classNames[ gtLabels[i] ],
                                                                 gtMask[i]), color='red')

    CocoDSet.showInferResults(img, np.array(matchingBoxes), matchingLabels, matchingScores, False,
                              "Showing Ground Truth boxes")

# **********************************************************************************************************************
def testEvaluation(isTraining=False):
    # To test the evaluation, copy the "EvalTestData.npz" file to the same folder as this unit test code.
    myPrint('\nTesting Evaluation %s...'%('in Training mode ' if isTraining else ''), color='cyan')
    if os.path.exists("AnchorBoxes.npz")==False:
        myPrint('    Ground Truth test data not available (AnchorBoxes.npz). Skipping this test.', color='yellow')
        return

    myPrint('    Preparing test data ... ')
    rootDic = np.load("EvalTestData.npz", encoding='latin1', allow_pickle=True)
    detections = list( rootDic['detections'] )
    imageIds = list( rootDic['imageIds'] )

    testDs = CocoDSet('Test', batchSize=32, resolution=512, keepAr=False)

    resultsDic = {}
    imgIds = []
    for d in detections:
        imgId = d['image_id']
        classId = d['category_id']
        classId = CocoDSet.classId2Idx[classId]
        box = CocoDSet.p1SizeToP1P2(np.float32(d['bbox']))
        score = d['score']

        if imgId not in resultsDic:
            imgIds += [imgId]
            resultsDic[imgId] = (imgId, ([], [], []) )

        imgId, dt = resultsDic[imgId]
        resultsDic[imgId] = (imgId, (dt[0]+[classId], dt[1]+[box], dt[2]+[score]))

    results = []
    for id in imgIds:
        f, dt = resultsDic[id]
        results += [ (f, (np.int32(dt[0]), np.float32(dt[1]), np.float32(np.sort( dt[2] )[::-1])))  ]

    if isTraining:
        startTime = time.time()
        myPrint('    Now Evaluating in Training mode ... ')
        mAP = testDs.evaluate(results, isTraining=True, quiet=True)
        if ('%.3f'%mAP)=='0.258':
            myPrint('    Success! (mAP=%.3f, %.2f Seconds)'%(mAP,time.time()-startTime), color='green')
        else:
            myPrint('    Error: (mAP=%.3f expected 0.258)\n'%(mAP), color='red')
            exit(0)
    else:
        myPrint('    Now Evaluating ... ')
        startTime = time.time()
        ap, ap50, ap75, ar, resultsStrs = testDs.evaluate(results, quiet=True)
        #           ap[0]    ap[1]    ap[2]    ap[3]    ap50[0]  ap75[0]  ar[0]    ar[1]    ar[2]    ar[3]    ar[4]    ar[5]
        expected = ["0.258", "0.101", "0.300", "0.379", "0.475", "0.254", "0.381", "0.176", "0.431", "0.526", "0.234", "0.356"]
        actual = ['%.3f'%ap[0], '%.3f'%ap[1], '%.3f'%ap[2], '%.3f'%ap[3], '%.3f'%ap50[0], '%.3f'%ap75[0],
                  '%.3f'%ar[0], '%.3f'%ar[1], '%.3f'%ar[2], '%.3f'%ar[3], '%.3f'%ar[4], '%.3f'%ar[5]]
        for i in range(len(expected)):
            if expected[i] != actual[i]:
                myPrint( '    Error: Results do not match expected values.', color='red')
                myPrint( '    Expected Values:')
                myPrint( '      Average Precision (AP):')
                myPrint( '          IoU=0.50:0.95   Area: All      MaxDet: 100  = 0.258')
                myPrint( '          IoU=0.50        Area: All      MaxDet: 100  = 0.475')
                myPrint( '          IoU=0.75        Area: All      MaxDet: 100  = 0.254')
                myPrint( '          IoU=0.50:0.95   Area: Small    MaxDet: 100  = 0.101')
                myPrint( '          IoU=0.50:0.95   Area: Medium   MaxDet: 100  = 0.300')
                myPrint( '          IoU=0.50:0.95   Area: Large    MaxDet: 100  = 0.379')
                myPrint( '      Average Recall (AR):')
                myPrint( '          IoU=0.50:0.95   Area: All      MaxDet: 1    = 0.234')
                myPrint( '          IoU=0.50:0.95   Area: All      MaxDet: 10   = 0.356')
                myPrint( '          IoU=0.50:0.95   Area: All      MaxDet: 100  = 0.381')
                myPrint( '          IoU=0.50:0.95   Area: Small    MaxDet: 100  = 0.176')
                myPrint( '          IoU=0.50:0.95   Area: Medium   MaxDet: 100  = 0.431')
                myPrint( '          IoU=0.50:0.95   Area: Large    MaxDet: 100  = 0.526')
                myPrint( '\n    Actual Results:')
                for line in resultsStrs: print('      '+ line)
                print('')
                exit(0)
        myPrint('    Success! (%.2f Seconds)'%(time.time()-startTime), color='green')

# **********************************************************************************************************************
def testDisp():
    # The following requires display functionality. (Should be run on Mac)
    # To enable this test, copy the "cat_dog.jpg" file to the same folder as this unit test code.
    myPrint('\nTesting the Display functionality (Interactive) ... ', color='cyan')
    if os.path.exists("cat_dog.jpg")==False:
        myPrint('    Ground Truth test image not available (cat_dog.jpg). Skipping this test.', color='yellow')
        return

    import cv2
    img = np.float32(cv2.imread("cat_dog.jpg")) # BGR format
    # The Ground-Truth Boxes (in P1P2 format)
    dogBox = np.float32([54, 176, 756, 588])
    catBox = np.float32([674, 260, 938, 569])
    boxes = np.float32([ dogBox, catBox ])
    boxes = CocoDSet.p1P2ToP1Size(boxes)
    scores = [.6, .8]
    labels = [1, 2]
    CocoDSet.classNames = ['Back', 'Dog', 'Cat']

    try:
        # Tesing original sample:
        myPrint('    Close the image windows to go to the next step.')
        CocoDSet.showImageAndBoxes(img, boxes, labels, "Original Sample")

        # Scaling to 512x512 (keeping aspect ratio)
        scaledImg, scaledBoxes, imgOrgSize = CocoDSet.scaleImageAndBoxes(img, boxes, 512, True)
        CocoDSet.showImageAndBoxes(scaledImg, scaledBoxes, labels, "Scaled to 512x512 keeping aspect ratio")

        # Flipping the image
        flippedImg, flippedBoxes = CocoDSet.flipImageHorizontally(scaledImg, scaledBoxes)
        CocoDSet.showImageAndBoxes(flippedImg, flippedBoxes, labels, "Flipped Horizontally")

        # Zooming out and moving
        zoomedImg, zoomedBoxes = CocoDSet.zoomOutAndMove(scaledImg, scaledBoxes, 400, 512, (100,100))
        CocoDSet.showImageAndBoxes(zoomedImg, zoomedBoxes, labels, "Zoomed out (400) and moved by 100,100")

        # Testing Inference results (keeping aspect ratio)
        boxes01 = scaledBoxes/512.0 # Change boxes to [0..1]
        CocoDSet.showInferResults(img, boxes01, labels, scores, True, "Showing inference results")

        # Scaling to 512x512 (not keeping aspect ratio)
        scaledImg, scaledBoxes, imgOrgSize = CocoDSet.scaleImageAndBoxes(img, boxes, 512, False)
        CocoDSet.showImageAndBoxes(scaledImg, scaledBoxes, labels, "Scaled to 512x512 not keeping aspect ratio")

        # Testing Inference results (not keeping aspect ratio)
        boxes01 = scaledBoxes/512.0 # Change boxes to [0..1]
        CocoDSet.showInferResults(img, boxes01, labels, scores, False, "Showing inference results (not keeping aspect ratio)")
        
    except:
        myPrint('    No Display Capability! Skipping this test.', color='yellow')
    
    CocoDSet.classNames = None

# **********************************************************************************************************************
def testRandomMutation():
    # The following requires display functionality. (Should be run on Mac)
    # To enable this test, copy the "cat_dog.jpg" file to the same folder as this unit test code.
    myPrint('\nTesting random mutation (Interactive) ... ', color='cyan')
    if os.path.exists("cat_dog.jpg")==False:
        myPrint('    Ground Truth test image not available (cat_dog.jpg). Skipping this test.', color='yellow')
        return

    import cv2
    img = np.float32(cv2.imread("cat_dog.jpg")) # BGR format
    # The Ground-Truth Boxes (in P1P2 format)
    dogBox = np.float32([54, 176, 756, 588])
    catBox = np.float32([674, 260, 938, 569])
    boxes = np.float32([ dogBox, catBox ])
    boxes = CocoDSet.p1P2ToP1Size(boxes)
    scores = [.6, .8]
    labels = [1, 2]

    try:
        # Testing Random Mutation:
        db = CocoDSet('Test')
        CocoDSet.classNames = ['Back', 'Dog', 'Cat']    # Override the class names for this test
        
        for i in range(20):
            myPrint('    Random Mutation %d'%(i+1))
            scaledImg, scaledBoxes, imgOrgSize = CocoDSet.scaleImageAndBoxes(img, boxes, 512, True)
            mutatedImg, mutatedBoxes = db.randomMutate(scaledImg, scaledBoxes)
            CocoDSet.showImageAndBoxes(mutatedImg, mutatedBoxes, labels, 'Random Mutation %d'%(i+1))
    except:
        myPrint('    No Display Capability! Skipping this test.', color='yellow')

    CocoDSet.classNames = None

# **********************************************************************************************************************
def testGroundTruth(batchSize=64, w=4):
    myPrint('\nTesting Ground Truth (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
    if os.path.exists("AnchorBoxes.npz") == False:
        myPrint('    Ground Truth test data not available (AnchorBoxes.npz). Skipping this test.', color='yellow')
        return
        
    rootDic = np.load("AnchorBoxes.npz", encoding='latin1')
    anchorBoxes = rootDic['AnchorBoxes']
    trainDs = CocoDSet.makeDatasets('Test', numWorkers=w)
    trainDs.setAcnchorBoxes(anchorBoxes)

    testBatches(trainDs, batchSize)

# **********************************************************************************************************************
def testIou():
    myPrint('\nTesting IoU calculations ...', color='cyan')
    startTime = time.time()
    dtBoxes = np.float32([[100,100,200,200]])
    gtBoxes = np.float32([[100,100,200,200], [80,80,220,220], [120,120,180,180], [50,120,250,180], [120,50,180,250],
                          [100,200,200,300], [220,220,250,250]])

    expectedResults = [[1., 0.5102041, 0.36, 0.375, 0.375, 0., 0.]]
    myPrint('    Testing without crowd ...')
    if np.allclose( CocoDSet.getIou(dtBoxes, gtBoxes, None), expectedResults ) == False:
        myPrint('    Error: Results do not match expected values!\n', color='red')
        exit(0)

    myPrint('    Testing with crowd ...')
    crowd = [0, 0, 0, 1, 0, 0, 0]
    expectedResults = [[1., 0.5102041, 0.36, 0.6, 0.375, 0., 0.]]
    if np.allclose( CocoDSet.getIou(dtBoxes, gtBoxes, crowd), expectedResults ) == False:
        myPrint('    Error: Results do not match expected values!\n', color='red')
        exit(0)
    myPrint('    Success! (%.2f Seconds)'%(time.time()-startTime), color='green')
    
# **********************************************************************************************************************
if __name__ == '__main__':
    testIou()
    testGroundTruth()
    testAnchorBoxes()
    testRandomMutation()
    testDisp()
    testEvaluation(isTraining=True)
    testEvaluation()
    
    batchSize=64
    for w in [0,8]:
        myPrint('\nTesting only test set (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
        testDs = CocoDSet.makeDatasets('Test', numWorkers=w)
        testBatches(testDs, batchSize)

    batchSize=1
    w=4
    myPrint('\nTesting one by one (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
    testDs = CocoDSet.makeDatasets('Test', numWorkers=w)
    testBatches(testDs, batchSize)

    batchSize=64
    w=8
    myPrint('\nTesting all datasets (batchSize=%d, numWorkers=%d) ...'%(batchSize, w), color='cyan')
    trainDs, testDs, validDs = CocoDSet.makeDatasets(numWorkers=w)
    CocoDSet.printStats(trainDs, testDs, validDs)

    testBatches(trainDs, batchSize)
    testBatches(testDs, batchSize)
    testBatches(validDs, batchSize)
