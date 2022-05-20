# Copyright (c) 2019-2021 InterDigital AI Research Lab
"""
This is an implementation of codebook quantization using a special type of KMeans algorithm (oneShotKMeans).
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 01/27/2021    Shahab Hamidi-Rad       Created the file using the latest code from InterDigital's MPEG NNR
#                                       contributions.
# **********************************************************************************************************************
import numpy as np
import multiprocessing
import time
import sys, os
from copy import copy

import fireball
from .printutils import myPrint
from .utils import *

# **********************************************************************************************************************
# Quantization Default values:
KMEANS_ITERS = 1
REUSE_EMPTY_CLUSTERS = False

# **********************************************************************************************************************
def getMse(tensor1, tensor2): return np.square( tensor1 - tensor2 ).mean()
    
# **********************************************************************************************************************
def classifyPoints(points, centroids):
    dists = np.abs( points.reshape((-1,1))-np.repeat(centroids.reshape((1,-1)),points.shape[0],0) )
    return np.argmin(dists, 1)

# **********************************************************************************************************************
def moveCenters(points, pointClusters, centroids):
    newCentroids = []
    memberCounts = []
    errors = []
    splitCenters = []
    
    for i,centroid in enumerate(centroids):
        pointsIdxInThisCluster = np.where(pointClusters==i)[0]
        if pointsIdxInThisCluster.shape[0]>0:
            pointsInThisCluster = points[pointsIdxInThisCluster]
            newCentroid = pointsInThisCluster.mean()
            newCentroids += [ newCentroid ]
            errors += [ getMse(pointsInThisCluster,newCentroid) ]
            splitCenters += [ [ (99.0*newCentroid+pointsInThisCluster.min())/100.0,
                                (99.0*newCentroid+pointsInThisCluster.max())/100.0 ] ]
        else:
            # Empty clusters do not move
            newCentroids += [ centroid ]
            errors += [ 0.0 ]
            splitCenters += [ [0.0, 0.0] ]

        memberCounts += [ pointsIdxInThisCluster.shape[0] ]
    
    return np.float32(newCentroids), np.int32(memberCounts), errors, splitCenters

# **********************************************************************************************************************
def oneShotKMeans(points, initCentroids, reuseEmptyClusters=False):
    # In One-shut K-Means we the number of iteration for K-Means is 1. (unless we want to reuse any empty clusters)
    pointClusters = classifyPoints(points, initCentroids)
    centroids, memberCounts, errors, splitCenters = moveCenters(points, pointClusters, initCentroids)
    
    for i in range(10):
        # "centroids" are the centers that we want. But we need to calculate the membership and
        # memberCounts for these centers.
        pointClusters = classifyPoints(points, centroids)
        newCentroids, memberCounts, errors, splitCenters = moveCenters(points, pointClusters, centroids)

        # If we don't care about empty cluster, we are done.
        if not reuseEmptyClusters:      break
        
        emptyClusterIndexes = np.where(memberCounts==0)[0]
        
        # If there are no empty clusters, we are done.
        if len(emptyClusterIndexes)==0: break

        # It should not take more than 2-3 times to split large clusters to fill empty clusters
        assert i<8, "Still trying to fill empty clusters after 8 tries!"

        # Now we want to find the worst clusters and split them until there is no more empty cluster.
        worstClusters = np.argsort(errors)[::-1]  # Index of cluster with largest error comes first
        w = 0
        for emptyClusterIndex in emptyClusterIndexes:
            worstClusterIndex = worstClusters[w]
            w += 1
            
            if errors[ worstClusterIndex ]==0: break    # No more clusters to split!
            
            # Split the worst cluster:
            newCentroids[ emptyClusterIndex ] = splitCenters[ worstClusterIndex ][ 0 ]
            newCentroids[ worstClusterIndex ] = splitCenters[ worstClusterIndex ][ 1 ]
       
        centroids = newCentroids
       
    return centroids, memberCounts, pointClusters

# **********************************************************************************************************************
def getCenters(pdf, pdfFactor, uniformCenters, numClusters):
    if pdfFactor == 0:  return uniformCenters
    
    # Clip the PDF based on pdfFactor
    numSteps = numClusters-1
    numPdfSteps = numSteps*16
    pdfStep = np.float32(uniformCenters[-1]-uniformCenters[0])/numPdfSteps
    lowerBound = pdf.mean()*(1.0 - pdfFactor)
    upperBound = pdf.mean()*(1.0 + pdfFactor)
    boundedPdf = np.clip(pdf, lowerBound, upperBound)

    cdf = np.cumsum(boundedPdf)
    cdfStep = cdf[-1]/numSteps
    intCenters = [ (np.searchsorted(cdf, center)) for center in np.arange(numClusters)*cdfStep ]
    centers = np.float32(intCenters)*pdfStep + uniformCenters[0]
    return centers

# **********************************************************************************************************************
def getEntropy(indexes, orgCounts=None, fast=False):
    entropy = 0     # Total number of bits for indexes
    if indexes.max() > 0:
        adaptive = 1 if orgCounts is None else -1
        if adaptive==1:
            countSum = indexes.max()+1
            counts = np.ones(countSum)
        else:
            counts = np.abs(orgCounts.copy())       # Do not corrupt the original counts
            countSum = counts.sum()
            if fast:
                probs = counts[indexes]/countSum
                return (-np.log2(probs)).sum()

        indexesList = indexes.flatten().tolist()
        for idx in indexesList:
            p = np.float32(counts[idx])/countSum      # Probability of idx
            entropy += -np.log2(p)                  # Entropy: Low prob. => less number of bits
            counts[idx] += adaptive
            countSum += adaptive

    return entropy

# **********************************************************************************************************************
def quantizeNoCodebook(dataArray, symCount):
    tensorMin = dataArray.min()
    tensorMax = dataArray.max()
    rangeInt = np.int32(np.ceil(tensorMax) - np.floor(tensorMin))
    if rangeInt == 0:
        codebookType = 1    # All data elements are the same integer value.
        return None, np.int32([codebookType, tensorMin]), 0

    step = np.float32(rangeInt/(symCount-1))
    offset = np.round(tensorMin/step)       # Note that this can be negative
    maxIndex = np.round(tensorMax/step)

    mse = getMse(dataArray, np.round(dataArray/step)*step)
    if offset == maxIndex:
        codebookType = 2    # All quantized dataArray elements are the same float value.
        return None, np.int32([codebookType, rangeInt, symCount, offset]), mse

    # Similar to Uniform quantization
    codebookType = 3
    indexes = np.uint32(np.round(dataArray/step)-offset)
    return indexes, np.int32([codebookType, rangeInt, symCount, offset]), mse

# **********************************************************************************************************************
def quantizeCodebook(dataArray, symCount, **kwargs):
    assert ((type(dataArray)==np.ndarray) and (len(dataArray.shape)==1) and (dataArray.dtype==np.float32)), \
           "Only 1-D numpy arrays of type 'float32' are accepted for quantization!"

    # By default, assume we want to re-train after quantization:
    trainedQuantization = kwargs.get('trainedQuantization', True)
    reuseEmptyClusters = kwargs.get('reuseEmptyClusters', True)     # Needed for CoreML
    
    tensorMin = dataArray.min()
    tensorMax = dataArray.max()
    rangeInt = np.int32(np.ceil(tensorMax) - np.floor(tensorMin))
    if rangeInt == 0:
        if trainedQuantization: return None, None, None, None
        # If not training after quantization, this helps improve compression
        codebookType = 1    # All tensor elements are the same integer value.
        return None, np.int32([codebookType, tensorMin]), None, 0

    assert symCount <= dataArray.size

    tensorRange = tensorMax-tensorMin

    if tensorRange == 0:
        if trainedQuantization: return None, None, None, None
        
        # If not training after quantization, this helps improve compression
        codebookType = 2    # All quantized tensor elements are the same float value.
        step = np.float32(rangeInt/(symCount-1))
        offset = np.round(tensorMin/step)       # Note that this can be negative
        maxIndex = np.round(tensorMax/step)
        mse = getError(tensor, np.round(tensor/step)*step)
        return None, np.int32([codebookType, rangeInt, symCount, offset]), None, mse
        
    # Uniform Initialization:
    numSteps = symCount-1
    uniformStep = np.float32(tensorRange)/numSteps
    assert uniformStep > 0
    uniformCenters = np.arange(symCount, dtype=np.float32)*uniformStep + np.round(tensorMin/uniformStep)*uniformStep
    initCenters = uniformCenters

    # Calculate PDF:
    numPdfSteps = numSteps<<4
    pdfStep = np.float32(uniformCenters[-1]-uniformCenters[0])/numPdfSteps
    pdfInts = np.round(dataArray/pdfStep).astype(np.int32)
    intVals, memCounts = np.unique(pdfInts, return_counts=True)
    pdf = np.zeros(numPdfSteps+1, dtype=np.float32)
    pdfOffset = int(uniformCenters[0]/pdfStep)
    for i,intVal in enumerate(intVals):
        if intVal<pdfOffset:                    pdf[ 0 ] += memCounts[i]
        elif (intVal-pdfOffset)>numPdfSteps:    pdf[ numPdfSteps ] += memCounts[i]
        else:                                   pdf[ intVal-pdfOffset ] += memCounts[i]
    
    pdfFactor = kwargs.get('pdfFactor', 0.1)
    if kwargs.get('maximizeEntropy', False):
        assert not trainedQuantization, "'maximizeEntropy' not supported for trained quantization!"
        # If we are not doing trained quantization, and 'maximizeEntropy' is specified, we find the best value for
        # 'pdfFactor' that maximizes the entropy.
        bestEntropy = np.inf
        bestPdf = 0
        for i in range(10):
            pdfFactor = i*.1
            
            initCenters = getCenters(pdf, pdfFactor, uniformCenters, symCount)
            centers, memCounts, membership = oneShotKMeans(dataArray, initCenters, reuseEmptyClusters)
            
            entropy = getEntropy(membership, memCounts, True)
            if entropy < bestEntropy:
                bestPdf = pdfFactor
                bestEntropy = entropy
                bestCenters, bestMemCounts, bestIndexes = centers, memCounts, membership
        centers, memCounts, membership = bestCenters, bestMemCounts, bestIndexes
    
    else:
        initCenters = getCenters(pdf, pdfFactor, uniformCenters, symCount)
        centers, memCounts, membership = oneShotKMeans(dataArray, initCenters, reuseEmptyClusters)
        
    # Sort Clusters based on their member count:
    sortOrder = np.argsort(memCounts)[::-1]     # Highest count first
    centers = centers[sortOrder]
    memCounts = memCounts[sortOrder]
    invSortOrder = np.argsort(sortOrder)
    indexes = invSortOrder[membership]

    if not reuseEmptyClusters:
        # Remove Empty Clusters
        numNonEmptyClusters = 0
        for count in memCounts:
            if count != 0:  numNonEmptyClusters += 1
            else:           break
        memCounts = memCounts[:numNonEmptyClusters]
        codebook = centers[:numNonEmptyClusters]
    else:
        codebook = centers
        codebookSize = len(centers)
        assert codebookSize==symCount, \
               "codebookSize (%d) must be equal to original symCount (%d) when \
               'reuseEmptyClusters' is True!"%(codebookSize,symCount)

    if trainedQuantization:
        return np.uint32(indexes), np.float32(codebook), memCounts, getMse(dataArray, codebook[indexes])

    # If we are not doing trained quantization, we quantize the codebook using linear quantization
    codebookMin = codebook.min()
    codebookMax = codebook.max()
    rangeInt = np.ceil(codebookMax) - np.floor(codebookMin)
    
    cbSymCount = max(1<<20, symCount*symCount)
    cbStep = np.float32(rangeInt/(cbSymCount-1))
    cbOffset = np.round(codebookMin/cbStep)
    quantCodebook = np.round(codebook/cbStep)*cbStep
    mse = getMse(dataArray, quantCodebook[indexes])

    codebookType = 0    # Quantized Codebook
    intCodebook = np.round(codebook/cbStep)-cbOffset
    codebookInfo = np.int32([0, rangeInt, cbSymCount, cbOffset ] + intCodebook.tolist())
    return np.uint32(indexes), codebookInfo, memCounts, mse

# **********************************************************************************************************************
def binSearchSymCount(symCountLo, symCountHi, target, func):
    symCount = symCountLo
    lo , hi = None, None
    loMse, hiMse = None, None
    while True:
        x = func(symCount)
        mse = x[-1]
        if mse is None: return x + tuple([None])      # No quantization
        if lo is None:
            lo, loMse = symCount, mse
            if mse < target:    break   # Got lucky!
            symCount *= 4
            if symCount > symCountHi:   symCount = symCountHi
            continue
            
        if hi is None:
            if mse > target:
                lo, loMse = symCount, mse
                if symCount == symCountHi:  break
                symCount *= 4
                if symCount > symCountHi:   symCount = symCountHi
                continue
                
            hi, hiMse = symCount, mse
            symCount = (hi + lo)//2
            continue

        if mse > target:
            lo, loMse = symCount, mse
            if (hi - lo) < 2:
                symCount = hi
                x = func(symCount)
                break
                
            symCount = (hi + lo)//2
            continue
            
        hi, hiMse = symCount, mse
        if (hi - lo) < 2:   break

        symCount = (hi + lo)//2
      
    return x + (symCount,)

# **********************************************************************************************************************
def quantize1dArray(dataArray, **kwargs):
    mseUb = kwargs.get('mseUb', None)
    assert mseUb is not None
    
    trainedQuantization = kwargs.get('trainedQuantization', True)
    reuseEmptyClusters = kwargs.get('reuseEmptyClusters', True)         # Needed for CoreML
        
    if reuseEmptyClusters:
        # In this case symCount must be a power of 2. Try 64 and decide to go up or down best on results
        minBits = kwargs.get('minBits', 2)
        maxBits = kwargs.get('maxBits', 12)
        startBits = (maxBits + minBits)//2
        assert (minBits <= maxBits), "'minBits' cannot be larger than 'maxBits'!"

        # If the dataArray is pruned, we reserve the first cluster for codebook value 0.0.
        numReservedClusters = 1 if kwargs.get('reserve0cluster', False) else 0
        
        symCount = (1<<startBits) - numReservedClusters
        indexes, codebookInfo, memcounts, mse = quantizeCodebook(dataArray, symCount, **kwargs)
        if mse is None:             return None, None, None, None   # Cannot quantize and mseUb is not the problem
        
        if mse > mseUb:
            for p in range(startBits+1, maxBits):
                symCount = (1<<p) - numReservedClusters
                indexes, codebookInfo, memcounts, mse = quantizeCodebook(dataArray, symCount, **kwargs)
                if mse <= mseUb:     break
                
        elif mse < mseUb:
            maxMse = mse
            for p in range(startBits-1, minBits,-1):
                symCount = (1<<p) - numReservedClusters
                testResults = quantizeCodebook(dataArray, symCount, **kwargs)
                if testResults[-1] > mseUb:     break
                indexes, codebookInfo, memcounts, mse = testResults
                    
        if trainedQuantization:
            if mse <= mseUb:  return indexes, codebookInfo, memcounts, mse
            return None, None, None, mse    # Cannot quantize because mseUb is too small, need more quantization bits
            
        codebookResults = indexes, codebookInfo, memcounts, mse
    else:
        codebookResults = None
        minSymCount = min(kwargs.get('minSymCount', 4), dataArray.size)
        maxSymCount = min(kwargs.get('maxSymCount', 4096), dataArray.size)
        
        # Do binary search to find best codebookSize (symCount)
        f = lambda x: quantizeCodebook(dataArray, x, **kwargs)
        cbIndexes, codebookInfo, memCounts, cbMse, cbSymCount = binSearchSymCount(minSymCount, maxSymCount, mseUb, f)
        
        if trainedQuantization:
            if cbMse is not None:
                if cbMse <= mseUb:  return cbIndexes, codebookInfo, memCounts, cbMse
            return None, None, None, None
            
        if cbMse <= mseUb:
            codebookResults = cbIndexes, codebookInfo, memCounts, cbMse
    
    assert not trainedQuantization
    # If not doing trained quantization, we try linear quantization and choose the one with best results.
    uniMseFactor = kwargs.get('uniMseFactor', 8.0)
    minSymCount = kwargs.get('minSymCountLinear', 4)
    maxSymCount = kwargs.get('maxSymCountLinear', 1<<28)
    
    # Do binary search to find best symCount
    f = lambda x: quantizeNoCodebook(dataArray, x)
    uniIndexes, uniInfo, uniMse, uniSymCount = binSearchSymCount(minSymCount, maxSymCount, maxMse/uniMseFactor, f)
    
    if codebookResults is None:
        return uniIndexes, uniInfo, None, uniMse
        
    if cbMse < uniMse:
        # MSE of codebook quantization is smaller than that of uniform which is supposed to
        # be "uniMseFactor" times smaller. This means codebook is not quantizing efficiently.
        return uniIndexes, uniInfo, None, uniMse

    if dataArray.size/codebookInfo.size < cbSizeRatio:
        return uniIndexes, uniInfo, None, uniMse

    return codebookResults

# **********************************************************************************************************************
def dequantize(indexesOrShape, codebookInfo):
    codebookType = codebookInfo[0]
    if codebookType == 1:
        # All same int value. "indexesOrShape" is the shape of tensor
        return np.ones(indexesOrShape, dtype=np.float32)*codebookInfo[1]
    
    _, rangeInt, symCount, offset = codebookInfo[:4]
    step = np.float32(rangeInt)/(symCount-1)

    if codebookType == 2:
        # All same float value. "indexesOrShape" is the shape of tensor
        return np.ones(indexesOrShape, dtype=np.float32)*offset*step
    
    if codebookType == 3:
        # No codebook (Uniform Quantization). "indexesOrShape" contains the integer values.
        return np.float32( (indexesOrShape+offset)*step )

    # There is a codebook. "indexesOrShape" contains the integer index values. First de-quantize the codebook
    # and then use it to lookup values.
    assert codebookType == 0
    assert len(codebookInfo) > 4
    quantizedCodebook = codebookInfo[4:]
    codebook = np.float32( (quantizedCodebook+offset)*step )
    return codebook[indexesOrShape]
