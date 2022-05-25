# Copyright (c) 2019-2021 InterDigital AI Research Lab
"""
This file contains the implementation for NetParam object.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 02/04/2021    Shahab Hamidi-Rad       Created the file. Moved the original implementation from the "layers.py" file
#                                       to this file.
# 05/12/2022    Shahab                  Adding support for calculating and maintaining the number of bytes for NetParam
#                                       objects.
# **********************************************************************************************************************
import numpy as np
import time

import tensorflow as tf
try:    import tensorflow.compat.v1 as tf1
except: tf1 = tf
try:    import tensorflow.random as tfr
except: tfr = tf
    
from .printutils import myPrint

# **********************************************************************************************************************
def quantizeWorker(netParam, kwargs):   return netParam.quantize(**kwargs)
def pruneWorker(netParam, mseUb):       return netParam.prune(mseUb)

# **********************************************************************************************************************
class NetParam:
    # NetParam encapsulates a single network parameter.
    # A NetParam can operate in 2 different modes:
    #   Mode TF: TensorFlow parameter (For example when used in a TensorFlow graph for training or inference)
    #   Mode NP: Numpy parameter (For example when saving/loading to files)
    # Regardless of the mode, a NetParam can also be:
    #   Quantized: In this case codebook is not None
    #   Pruned: In this case bitMask is not None
    
    # ******************************************************************************************************************
    def __init__(self, mode, rawVal, codebook=None, bitMask=None, trainable=True, name=None):
        assert type(mode)==str
        assert mode in ['NP', 'TF']
        assert rawVal is not None
        
        self.mode = mode
        self.rawVal = rawVal
        self.codebook = codebook
        self.bitMask = bitMask
        self.trainable = trainable
        self.tfValue = None
        self.initialized = False
        self.name = name
                    
        if self.mode == 'NP':
            madeUpName = "NetParam"
            assert type(self.rawVal)==np.ndarray
            if self.bitMask is not None:
                madeUpName += "-Sparse"
                assert type(self.bitMask) == np.ndarray
                assert bitMask.dtype == np.uint8
        
            bytesPerVal = 4
            if self.codebook is not None:
                madeUpName += "-Quantized"
                assert self.codebook.dtype == np.float32, str(self.codebook.dtype)+"?"
                assert self.rawVal.dtype in [np.int32, np.uint32, np.uint16, np.uint8], str(self.rawVal.dtype)+"?"
                if len(self.codebook)<=256:
                    self.rawVal = np.uint8(self.rawVal)
                    bytesPerVal = 1
                else:
                    self.rawVal = np.uint16(self.rawVal)
                    bytesPerVal = 2
            
            madeUpName += "-" + "x".join([str(x) for x in self.rawVal.shape])
            if self.name is None:   self.name = madeUpName
            
            self.size = self.rawVal.size
            self.shape = self.rawVal.shape
            self.numParams = int(self.size if self.bitMask is None else self.bitMask.sum())

            self.numBytesRam = self.size * 4
            if codebook is not None:        self.numBytesRam += 4*len(self.codebook)
            if self.bitMask is not None:    self.numBytesRam += self.numParams
            self.numBytesRam += 4       # Additional overhead for shape and flags

            # numBytes is used for compression optimization purposes
            self.numBytes = self.numParams * bytesPerVal
            if codebook is not None:        self.numBytes += 4*len(self.codebook)
            if self.bitMask is not None:    self.numBytes += 4*int(np.ceil(self.size/32.0))

            self.numBytesDisk = self.numParams * bytesPerVal
            if codebook is not None:        self.numBytesDisk += 4*len(self.codebook)
            if self.bitMask is not None:
                # The overhead includes: flags, shape info, and compressed bitMasks
                self.numBytesDisk += 1
                self.numBytesDisk += len(self.shape) if np.any(np.uint32(self.shape)>0xFFFF) else (len(self.shape)+1)//2
                self.numBytesDisk += 4*int(np.ceil(self.size/32.0))
            self.numBytesDisk += 4      # Additional overhead for shape and flags

        else:
            self.rawVal = rawVal
            self.codebook = codebook
            self.bitMask = bitMask
            
            if not trainable:
                assert codebook is None
                assert bitMask is None
                self.tfValue = self.rawVal
                
            elif self.codebook is not None:
                if self.bitMask is None:
                    self.tfValue = tf1.gather(self.codebook, self.rawVal)
                else:
                    # In this case the first codebook item is related to the pruned zeros.
                    # This codebook entry must always remain 0 during the training (not trained)
                    # The following codebookMask ensures the actual lookup table always returns 0 for the
                    # Index 0.
                    codebookMask = tf1.constant([0] + [1]*(self.codebookSize-1), dtype=tf.float32)
                    self.tfValue = tf1.gather(self.codebook*codebookMask, self.rawVal)
                    
            elif self.bitMask is not None:
                self.tfValue = self.rawVal * tf.cast(self.bitMask, tf.float32)
                
            else:
                self.tfValue = self.rawVal

    # ******************************************************************************************************************
    def __repr__(self):
        """
        __repr__
        Returns a text string briefly describing this instance of "NetParam".
        """
        retStr = '\n%s Instance:'%(self.__class__.__name__)
        retStr += '\n    %s: %s'%('name', self.name)
        retStr += '\n    %s: %s'%('mode', self.mode)
        retStr += '\n    %s: %s'%('shape', 'x'.join(str(x) for x in self.shape))
        retStr += '\n    %s: %s'%('size', str(self.size))
        retStr += '\n    %s: %s'%('trainable', "Yes" if self.trainable else "No")
        retStr += '\n    %s: %s'%('numParams', str(self.numParams))
        retStr += '\n    %s: %s'%('Pruned', 'No' if self.bitMask is None else 'Yes')
        if self.bitMask is not None:
            retStr += '\n    %s: %s'%('numPruned', str(self.numPruned))
        retStr += '\n    %s: %s'%('Quantized', 'No' if self.codebook is None else 'Yes')
        if self.codebook is not None:
            retStr += '\n    %s: %s'%('codebookSize', str(self.codebookSize))
        retStr += '\n    %s: %s'%('numBytesRam', str(self.numBytesRam))
        retStr += '\n    %s: %s'%('numBytesDisk', str(self.numBytesDisk))
        retStr += '\n    %s: %s'%('numBytes', str(self.numBytes))

        retStr += '\n    %s: %s'%('rawVal', 'x'.join(str(x) for x in self.rawVal.shape) + (" (%s)"%str(self.rawVal.dtype)))
        if self.bitMask is not None:
            retStr += '\n    %s: %s'%('bitMask', 'x'.join(str(x) for x in self.bitMask.shape) + (" (%s)"%str(self.bitMask.dtype)))
        if self.codebook is not None:
            retStr += '\n    %s: %s'%('codebook', 'x'.join(str(x) for x in self.codebook.shape) + (" (%s)"%str(self.codebook.dtype)))

        return retStr

    # ******************************************************************************************************************
    def setSizeInfo(self, npNetParam=None, shape=None):
        if npNetParam is not None:
            self.size = npNetParam.rawVal.size
            self.shape = npNetParam.shape
            self.numParams = npNetParam.numParams
            self.numBytesRam = npNetParam.numBytesRam
            self.numBytesDisk = npNetParam.numBytesDisk
            self.numBytes = npNetParam.numBytes
            self.initialized = True
        else:
            # This is when the parameter is made from scratch (Randomly or constant)
            # This NetParam cannot be a quantized or pruned parameter.
            # Assuming type is float 32
            assert shape is not None
            assert self.codebook is None
            assert self.bitMask is None
            
            self.shape = shape
            self.size = int(np.int32(shape).prod())
            self.numParams = self.size
            self.numBytesRam = self.size * 4
            self.numBytesDisk = self.size * 4
            self.numBytes = self.size * 4
            
    # ******************************************************************************************************************
    def packMasks(self):
        flatMask = self.bitMask.flatten()
        flatTensor = self.rawVal.flatten()
        mask32s = []    # BitMasks packed into a list of uint32s
        nzVals = []     # Non-Zero values in the rawVal
        MASKVALS = [1<<i for i in range(32)]
        curMask32 = 0
        for i,bit in enumerate(flatMask):
            if (i&0x1F)==0 and i!=0:
                mask32s += [curMask32]
                curMask32 = 0

            if bit==1:
                nzVals += [ flatTensor[i] ]
                curMask32 += MASKVALS[i%32]
        mask32s += [curMask32]
        
        return np.array(nzVals, dtype=self.rawVal.dtype), np.uint32(mask32s)

    # ******************************************************************************************************************
    @classmethod
    def unpackMasks(cls, nzVals, mask32s, shape):
        flatVals = []
        flatBitMask = []
        r = 0
        for mask32 in mask32s:
            bits = np.unpackbits(np.uint32([mask32]).view(np.uint8), bitorder='little').tolist()
            flatBitMask += bits
            for bit in bits:
                if bit==0:
                    flatVals += [0]
                else:
                    flatVals += [nzVals[r]]
                    r += 1
        l = np.int32(shape).prod()
        return np.array(flatVals[:l], dtype=nzVals.dtype).reshape(shape), np.uint8(flatBitMask[:l]).reshape(shape)

    # ******************************************************************************************************************
    def encodeSparseInfo(self):
        sparseInfo = []
        dimensionality = self.dim
        bigTensor = 4 if np.any(np.uint32(self.shape)>0xFFFF) else 0
        format = 0
        flags = (dimensionality-1) + bigTensor + format*8
        sparseInfo += [flags]
        
        for d,dim in enumerate(self.shape):
            if bigTensor!=0:   sparseInfo += [dim]
            else:
                if (d%2)==0:   sparseInfo += [dim&0xFFFF]
                else:          sparseInfo[-1] += (dim<<16)&0xFFFF0000
                    
        nzVals, mask32s = self.packMasks()
        sparseInfo += mask32s.tolist()
        
        return np.array(nzVals, dtype=self.rawVal.dtype), np.uint32(sparseInfo)

    # ******************************************************************************************************************
    @classmethod
    def decodeSparseInfo(cls, nzVals, sparseInfo):
        
        if sparseInfo is None:  return nzVals, None     # Not sparse
        assert sparseInfo.dtype==np.uint32
        flags = sparseInfo[0]
        dimensionality = (flags & 3) + 1
        bigTensor = (flags & 4) != 0
        format = {0:'Bitmap', 1:'CSR', 2:'Reserved', 3:'Reserved'}[ (flags//8)%4 ]
        assert format in ['Bitmap']

        shape = []
        for d in range(dimensionality):
            idx = 1 + d
            dim = sparseInfo[1 + d//(1 if bigTensor else 2)]
            if not bigTensor: dim = (dim if (d%2)==0 else (dim >> 16)) & 0xFFFF
            shape += [ dim ]

        maskIdx = (1 + dimensionality) if bigTensor else (1+(dimensionality+1)//2)
        masks = sparseInfo[ maskIdx : ]
        
        return cls.unpackMasks(nzVals, masks, shape)
        
    # ******************************************************************************************************************
    @classmethod
    def loadNetParams(cls, fileDic):
        if 'netParamDics' in fileDic:
            # loading a compressed model
            from .arithcoder import decodeNetParam
            netParams = cls.decompressNetParams( fileDic['netParamDics'] )
        else:
            rawVals = fileDic.get('netParams', None)
            quantInfo = fileDic.get('quantInfo', None)
            sparseInfo = fileDic.get('sparseInfo', None)
            flags = fileDic.get('netParamFlags', None)
            
            if rawVals is None: return None
            numParams = len(rawVals)

            if quantInfo is not None:   assert len(quantInfo)==numParams
            if sparseInfo is not None:  assert len(sparseInfo)==numParams
            if flags is not None:       assert len(flags)==numParams
            
            netParams = []  # Return a list of NetParam objects
            for i in range(numParams):
                rawVal, bitMask = NetParam.decodeSparseInfo(rawVals[i], None if sparseInfo is None else sparseInfo[i])
                codebook = None if quantInfo is None else quantInfo[i]
                trainable = (flags is None) or ((flags[i]&1) == 1)

                netParams += [ NetParam('NP', rawVal, codebook, bitMask, trainable) ]
        
        return netParams

    # ******************************************************************************************************************
    def getSaveInfo(self):
        assert self.mode == 'NP'
        flags = 1 if self.trainable else 0
        
        if self.bitMask is None:    return self.rawVal, self.codebook, None, flags
        nzVals, sparseInfo = self.encodeSparseInfo()
        return nzVals, self.codebook, sparseInfo, flags
        
    # ******************************************************************************************************************
    @classmethod
    def saveNetParams(cls, fileDic, netParams):
        rawVals = []
        quantInfo = []
        sparseInfo = []
        flags = []
        
        for netParam in netParams:
            rv, qi, si, f = netParam.getSaveInfo()
            rawVals += [ rv ]
            quantInfo += [ qi ]
            sparseInfo += [ si ]
            flags += [ f ]
        
        fileDic['netParams'] = rawVals
        fileDic['quantInfo'] = quantInfo
        fileDic['sparseInfo'] = sparseInfo
        fileDic['netParamFlags'] = flags

    # ******************************************************************************************************************
    @classmethod
    def pruneNetParams(cls, netParams, **kwargs):
        import multiprocessing

        quiet = kwargs.get('quiet', False)
        verbose = kwargs.get('verbose', True) and (not quiet)
        numWorkers = kwargs.get('numWorkers', None)
        minReductionPercent = kwargs.get('minReductionPercent', None)
        mseUb = kwargs['mseUb']
        n = len(netParams)
        
        if numWorkers is None:
            cpuCount = multiprocessing.cpu_count()
            numWorkers = max(min(cpuCount - 4, n), 0)
            
        if numWorkers > 0:
            processes = []
            pool = multiprocessing.Pool(numWorkers)
            if verbose: myPrint('Pruning %d tensors using %d workers ... '%(n, numWorkers))
        else:
            if verbose: myPrint('Pruning %d tensors ... '%(n))
        
        if verbose:
            myPrint('   Pruning Parameters:')
            myPrint('        mseUb ................ %f'%(mseUb))
            if minReductionPercent is not None:
                myPrint('        minReductionPercent .. %d%%'%(minReductionPercent))

        prunedNetParams = []
        totalPruned = 0
        orgNumParams = 0
        for p, netParam in enumerate(netParams):
            orgNumParams += netParam.numParams

            if numWorkers > 0:
                if netParam.codebook is not None:       processes += [None]
                elif netParam.bitMask is not None:      processes += [None]
                elif netParam.dim == 1:                 processes += [None]
                else:
                    processes += [ pool.apply_async(pruneWorker, (netParam, mseUb,)) ]
            else:
                if verbose:
                    s = '    Tensor %d of %d Shape: %s '%(p+1, n, 'x'.join(str(i) for i in netParam.shape))
                    myPrint(s + '.'*(45 - len(s)) + ' ', False)

                prunedNetParam = None
                if netParam.codebook is not None:
                    if verbose:   myPrint('Ignored. (Quantized Tensor)')
                elif netParam.bitMask is not None:
                    if verbose:   myPrint('Ignored. (Already pruned)')
                elif netParam.dim == 1:
                    if verbose:   myPrint('Ignored. (1-D Tensor)')
                else:
                    prunedNetParam, mse, nzmav = netParam.prune(mseUb)
                    
                    if verbose:
                        # print MSE with two decimal more than the mseUb
                        numDecimals = 2-int(np.floor(np.log10(mseUb)))
                        mseStr = ("%%.%df"% (numDecimals)) % (mse)

                    reductionPercent = 0
                    if prunedNetParam is None:
                        if verbose: myPrint('Not pruned! (mseUb too small - MSE:%s, nzmav:%s)'%(mseStr, str(nzmav)))
                    elif prunedNetParam.numPruned < int(np.ceil(netParam.size/32.0)):
                        if verbose: myPrint('Not pruned! (No Reduction - numPruned=%d)'%(prunedNetParam.numPruned))
                        prunedNetParam = None
                    elif minReductionPercent is not None:
                        newNumBytes = prunedNetParam.numBytes
                        orgNumBytes = netParam.numBytes
                        reductionPercent = (orgNumBytes-newNumBytes)*100.0/orgNumBytes
                        if reductionPercent < minReductionPercent:
                            if verbose: myPrint('Not pruned! (Small Reduction. %.1f%% < %d%%)'%(reductionPercent,
                                                                                                minReductionPercent))
                            prunedNetParam = None
                                                
                if prunedNetParam is None:
                    prunedNetParams += [ netParam ]
                else:
                    totalPruned += prunedNetParam.numPruned
                    prunedNetParams += [ prunedNetParam ]
                    if verbose: myPrint('Done. %d Pruned < %f, MSE=%s, Reduced: %.1f%%)'%(prunedNetParam.numPruned,
                                                                                          nzmav, mseStr,
                                                                                          reductionPercent))

            if (not quiet) and (not verbose):
                myPrint('    Processed %d tensor%s (of %d).\r'%(p+1, "" if p==0 else "s", n), False)

        if numWorkers > 0:
            numDone = 0
            while numDone<n:
                time.sleep(.1)
                numDone = 0
                for i in range(n):
                    if processes[i] is None:    numDone += 1; continue
                    try:
                        if processes[i].successful(): numDone += 1
                    except: pass
                if not quiet:
                    myPrint('    Processed %d tensor%s (of %d).\r'%(numDone, "" if numDone==1 else "s", n), False)
            
            pool.close()
            pool.join()

            for p,process in enumerate(processes):
                if process is not None:     prunedNetParam, mse, nzmav = process.get()
                else:                       prunedNetParam = None
                if prunedNetParam is not None:
                    if minReductionPercent is not None:
                        newNumBytes = prunedNetParam.numBytes
                        orgNumBytes = netParam.numBytes
                        reductionPercent = (orgNumBytes-newNumBytes)*100.0/orgNumBytes
                        if reductionPercent < minReductionPercent:
                            prunedNetParam = None

                if prunedNetParam is None:
                    prunedNetParams += [ netParams[p] ]
                else:
                    totalPruned += prunedNetParam.numPruned
                    prunedNetParams += [ prunedNetParam ]

        return prunedNetParams, orgNumParams, totalPruned

    # ******************************************************************************************************************
    @classmethod
    def quantizeNetParams(cls, netParams, **kwargs):
        import multiprocessing
        
        mseUb = kwargs['mseUb']
        quiet = kwargs.get('quiet', False)
        verbose = kwargs.get('verbose', True) and (not quiet)
        numWorkers = kwargs.get('numWorkers', None)
        weightsOnly = kwargs.get('weightsOnly', True)
        n = len(netParams)
        
        if numWorkers is None:
            cpuCount = multiprocessing.cpu_count()
            numWorkers = max(min(cpuCount - 4, n), 0)
            
        if numWorkers > 0:
            processes = []
            pool = multiprocessing.Pool(numWorkers)
            if verbose: myPrint('Quantizing %d tensors using %d workers ... '%(n, numWorkers))
        else:
            if verbose: myPrint('Quantizing %d tensors ... '%(n))
        
        if verbose:
            reuseEmptyClusters = kwargs.get('reuseEmptyClusters', True)     # Needed for CoreML
            myPrint('   Quantization Parameters:')
            myPrint('        mseUb .............. %s'%(mseUb))
            myPrint('        pdfFactor .......... %s'%(str(kwargs.get('pdfFactor', 0.1))))
            myPrint('        reuseEmptyClusters . %s'%(str(reuseEmptyClusters)))
            myPrint('        weightsOnly ........ %s'%(str(weightsOnly)))
            if reuseEmptyClusters:
                myPrint('        minBits ............ %s'%(str(kwargs.get('minBits', 2))))
                myPrint('        maxBits ............ %s'%(str(kwargs.get('maxBits', 12))))
            else:
                myPrint('        minSymCount ........ %s'%(str(kwargs.get('minSymCount', 4))))
                myPrint('        maxSymCount ........ %s'%(str(kwargs.get('maxSymCount', 4096))))

        quantizedNetParams = []
        originalBytes = 0
        quantizedBytes = 0
        
        for p, netParam in enumerate(netParams):
            originalBytes += netParam.numBytesDisk

            if numWorkers > 0:
                if weightsOnly and (netParam.dim==1):
                    processes += [None]
                elif netParam.codebook is not None:
                    processes += [None]
                else:
                    processes += [ pool.apply_async(quantizeWorker, (netParam, kwargs,)) ]
            else:
                if verbose:
                    s = '    Tensor %d of %d Shape: %s '%(p+1, n, 'x'.join(str(i) for i in netParam.shape))
                    myPrint(s + '.'*(45 - len(s)) + ' ', False)
               
                if weightsOnly and (netParam.dim==1):   quantizedNetParam = None
                elif netParam.codebook is not None:     quantizedNetParam = None
                else:                                   quantizedNetParam, mse = netParam.quantize(**kwargs)

                if verbose:
                    if mse is not None:
                        # print MSE with one decimal more than the mseUb
                        numDecimals = 1-int(np.floor(np.log10(mseUb)))
                        mseStr = ("%%.%df"% (numDecimals)) % (mse)

                    if weightsOnly and (netParam.dim==1):   myPrint('Ignored. (1-D Tensor)')
                    elif netParam.codebook is not None:     myPrint('Ignored. (Already quantized)')
                    elif quantizedNetParam is None:
                        if mse is None: myPrint('Not Quantized.')
                        else:           myPrint('Not Quantized. (MSE: %s)'%(mseStr) )
                    else:
                        myPrint('Quantized. (%s clusters - MSE: %s)'%(str(quantizedNetParam.codebookSize), mseStr) )
                
                if quantizedNetParam is None:   quantizedNetParam = netParam
                quantizedNetParams += [ quantizedNetParam ]
                quantizedBytes += quantizedNetParam.numBytesDisk
                
            if (not quiet) and (not verbose):
                myPrint('    Processed %d tensor%s (of %d).\r'%(p+1, "" if p==0 else "s", n), False)
                    
        if numWorkers > 0:
            numDone = 0
            while numDone<n:
                time.sleep(.1)
                numDone = 0
                for i in range(n):
                    if processes[i] is None:    numDone += 1; continue
                    try:
                        if processes[i].successful(): numDone += 1
                    except: pass
                if not quiet:
                    myPrint('    Processed %d tensor%s (of %d).\r'%(numDone, "" if numDone==1 else "s", n), False)
            
            pool.close()
            pool.join()

            for p,process in enumerate(processes):
                if process is not None:     quantizedNetParam, mse = process.get()
                else:                       quantizedNetParam = None
                quantizedNetParam = netParams[p] if quantizedNetParam is None else quantizedNetParam
                quantizedNetParams += [ quantizedNetParam ]
                quantizedBytes += quantizedNetParam.numBytesDisk

        return quantizedNetParams, originalBytes, quantizedBytes
        
    # ******************************************************************************************************************
    @classmethod
    def compressNetParams(cls, netParams, **kwargs):
        from .arithcoder import encodeNetParam
        import multiprocessing
        
        t0 = time.time()
        n = len(netParams)

        quiet = kwargs.get('quiet', False)
        verbose = kwargs.get('verbose', True) and (not quiet)
        numWorkers = kwargs.get('numWorkers', None)
        if numWorkers is None:
            cpuCount = multiprocessing.cpu_count()
            numWorkers = max(min(cpuCount - 4, n), 0)
            
        if numWorkers > 0:
            processes = []
            pool = multiprocessing.Pool(numWorkers)
            if verbose: myPrint('Compressing %d tensors using %d workers ... '%(n, numWorkers))
        else:
            if verbose: myPrint('Compressing %d tensors ... '%(n))
        
        netParamDics = []
        for p, netParam in enumerate(netParams):
            if numWorkers > 0:
                processes += [ pool.apply_async(encodeNetParam, (netParam,)) ]
            else:
                if verbose:
                    s = '    Tensor %d of %d Shape: %s '%(p+1, n, 'x'.join(str(i) for i in netParam.shape))
                    myPrint(s + '.'*(45 - len(s)) + ' ', False)
               
                netParamDic = encodeNetParam(netParam)

                if verbose:
                    newBytes = sum([value.size for key,value in netParamDic.items()])
                    myPrint('Compressed. (%d -> %d bytes)'%(netParam.numBytesDisk, newBytes) )
                
                netParamDics += [ netParamDic ]

            if (not quiet) and (not verbose):
                myPrint('    Compressed %d tensor%s (of %d).\r'%(p+1, "" if p==0 else "s", n), False)
                
        if numWorkers > 0:
            numDone = 0
            while numDone<n:
                time.sleep(.1)
                numDone = 0
                for i in range(n):
                    try:
                        if processes[i].successful(): numDone += 1
                    except: pass
                if not quiet:
                    myPrint('    Compressed %d tensor%s (of %d).\r'%(numDone, "" if numDone==1 else "s", n), False)
            
            pool.close()
            pool.join()

            for p,process in enumerate(processes):
                netParamDic = process.get()
                netParamDics += [ netParamDic ]
    
        return netParamDics

    # ******************************************************************************************************************
    @classmethod
    def decompressNetParams(cls, netParamDics):
        from .arithcoder import decodeNetParam
        import multiprocessing
        
        n = len(netParamDics)

        cpuCount = multiprocessing.cpu_count()
        numWorkers = max(min(cpuCount - 4, n), 0)

        if numWorkers > 0:
            processes = []
            pool = multiprocessing.Pool(numWorkers)
        
        myPrint("")
        netParams = []
        for p, netParamDic in enumerate(netParamDics):
            if numWorkers > 0:  processes += [ pool.apply_async(decodeNetParam, (netParamDic,)) ]
            else:
                myPrint('    Decompressing tensor %d of %d\r'%(p+1, n), False)
                netParams += [ decodeNetParam(netParamDic) ]
                
        if numWorkers > 0:
            numDone = 0
            while numDone<n:
                time.sleep(.1)
                numDone = 0
                for i in range(n):
                    try:
                        if processes[i].successful(): numDone += 1
                    except: pass
                myPrint('    Decompressed %d tensor%s (of %d)\r'%(numDone, "" if numDone==1 else "s", n), False)
            
            pool.close()
            pool.join()
            for p,process in enumerate(processes):  netParams += [ process.get() ]
        
        myPrint('    Decompressed %d tensors.              '%(n))
        return netParams
        
    # ******************************************************************************************************************
    def makeTf(self, name, trainable=True):
        # This function makes a new "TF" NetParam object from this "NP" NetParam
        assert self.mode=="NP"
        
        # If this function is called with "trainable=False", ignore initVal.trainable and use False.
        if trainable: trainable = self.trainable
        
        tfCodebook = None
        tfRawVal = None
        tfBitMask = None

        if not trainable:
            tfRawVal = tf1.get_variable(initializer=self.value(), name=name, trainable=False)
            
        else:
            if self.codebook is not None:
                tfRawVal = tf1.constant(self.rawVal, dtype=tf.int32, name='%sIndexes'%(name))
                tfCodebook = tf1.get_variable(initializer=self.codebook, name='%sCodeBook'%(name))
            else:
                tfRawVal = tf1.get_variable(initializer=self.rawVal, name=name)
            
            if self.bitMask is not None:
                tfBitMask = tf1.constant(self.bitMask, dtype=tf.uint8, name='%sBitMask'%(name))

        netParam = NetParam("TF", tfRawVal, tfCodebook, tfBitMask, trainable, name)
        netParam.setSizeInfo(self)  # Copy the size info from this NetParam
        
        return netParam
            
    # ******************************************************************************************************************
    @property
    def dim(self):
        return len(self.shape)

    # ******************************************************************************************************************
    @property
    def codebookSize(self):
        # Works for both tf and np
        if self.codebook is None: return 0
        return int(self.codebook.shape[0])

    # ******************************************************************************************************************
    @property
    def numPruned(self):
        return self.size-self.numParams
        
    # ******************************************************************************************************************
    @property
    def tfVariable(self):
        assert self.mode=="TF"
        if self.codebook is None:   return self.rawVal
        return self.codebook

    # ******************************************************************************************************************
    def tfL2Loss(self):
        return tf.nn.l2_loss(self.tfValue)
        
    # ******************************************************************************************************************
    def value(self, tfSession=None):
        if self.mode == 'NP':
            val = self.rawVal if self.codebook is None else self.codebook[self.rawVal]
            if self.bitMask is None: return val
            return val * self.bitMask
            
        if tfSession is None:
            return self.tfValue
            
        return tfSession.run(self.tfValue)
        
    # ******************************************************************************************************************
    @classmethod
    def toNpValues(cls, tfNetParams, tfSession):
        return tfSession.run([tfNetParam.tfValue for tfNetParam in tfNetParams])
        
    # ******************************************************************************************************************
    @classmethod
    def toNp(cls, tfNetParams, tfSession):
        # Converts a list of "TF" netParams to a list of "NP" netParams
        
        npParamInfo = [] # list of tuples (rawValIdx, codebookIdx, bitMaskIdx, trainable, name)
        i = 0
        allItemsToRun = []
        for tfNetParam in tfNetParams:
            assert tfNetParam.mode == "TF"
            itemsToRun = [ tfNetParam.rawVal, tfNetParam.codebook, tfNetParam.bitMask ]
            allItemsToRun += [x for x in itemsToRun if x is not None]
            
            indexes = [ i ]
            i += 1

            if tfNetParam.codebook is None:
                indexes += [None]
            else:
                indexes += [i]
                i+=1
                
            if tfNetParam.bitMask is None:
                indexes += [None]
            else:
                indexes += [i]
                i+=1
            
            npParamInfo += [ tuple(indexes + [tfNetParam.trainable, tfNetParam.name]) ]
        
        npInfo = tfSession.run(allItemsToRun)
        
        npNetParams = []
        for (rawValIdx, codebookIdx, bitMaskIdx, trainable, name) in npParamInfo:
            codebook = None
            if codebookIdx is not None:
                codebook = npInfo[codebookIdx]
                if bitMaskIdx is not None:
                    codebook *= np.int32([0]+[1]*(len(codebook)-1))
            npNetParams += [ NetParam('NP', npInfo[rawValIdx], codebook,
                                      None if bitMaskIdx is None else npInfo[bitMaskIdx],
                                      trainable, name) ]
        return npNetParams
        
    # ******************************************************************************************************************
    @classmethod
    def getAllTfVars(cls, netParams, trainable=None, initialized=None):
        # Returns a list of Variable objects in netParams (Used for gradients)
        tfVars = []
        for netParam in netParams:
            if (trainable == True) and (not netParam.trainable):        continue
            if (trainable == False) and (netParam.trainable):           continue
            if (initialized == True) and (not netParam.initialized):    continue
            if (initialized == False) and (netParam.initialized):       continue
            tfVars += [ netParam.tfVariable ]
        return tfVars

    # ******************************************************************************************************************
    def prune(self, mseUb):
        assert self.mode == 'NP'
        assert self.size>1
        assert self.bitMask is None
        assert self.codebook is None
        
        # We do a binary search for the Non-Zero Min Absolute Value (nzmav)
        nzmavLo, nzmavHi = 0.0, self.rawVal.max()+1
        npLo, npHi = 0, self.size+1
        mseLo = np.inf

        while (npHi-npLo)>1:
            if (nzmavHi-nzmavLo)<1e-12:    break
            nzmav = (nzmavLo + nzmavHi)/2.0
            nzMap = np.uint8(np.abs(self.rawVal)>nzmav)
            prunedTensor = self.rawVal * nzMap
            mse = np.square(prunedTensor-self.rawVal).mean()

            if mse > mseUb:     nzmavHi, npHi        = nzmav, int(self.size - nzMap.sum())
            else:               nzmavLo, npLo, mseLo = nzmav, int(self.size - nzMap.sum()), mse
                                        
        if nzmavLo != nzmav:
            nzmav = nzmavLo
            nzMap = np.uint8(np.abs(self.rawVal)>nzmav)
            numPruned = int(self.size - nzMap.sum())
            mse = min(mseLo,mse)
        else:
            numPruned = npLo
            mse = mseLo
        
        if (mse>mseUb) or (numPruned==0) :   return None, mse, nzmav
        
        prunedNetParam = NetParam('NP', self.rawVal*nzMap, None, nzMap, self.trainable,
                                  None if self.name[:8] == "NetParam" else self.name)

        return prunedNetParam, mse, nzmav

    # ******************************************************************************************************************
    def quantize(self, **kwargs):
        from .quantizer import quantize1dArray
        
        assert self.mode=='NP'
        kwargs['trainedQuantization'] = True
        
        if self.bitMask is None:
            indexes, codebook, memcounts, mse = quantize1dArray(self.rawVal.flatten(), **kwargs)
            if codebook is None:    return None, mse    # No Quantization
            
            quantizedNetParam = NetParam('NP', indexes.reshape(self.shape), codebook, None, self.trainable,
                                         None if self.name[:8] == "NetParam" else self.name)
            return quantizedNetParam, mse

        # This is a pruned NetParam. In this case:
        # 1) we quantize only non-zero values.
        # 2) If quantizing for CodeML ( reuseEmptyClusters==True), then reserve first cluster for 0
        if kwargs.get('reuseEmptyClusters', True):  kwargs['reserve0cluster'] = True
        x = self.rawVal * self.bitMask
        indexes, codebook, memcounts, mse = quantize1dArray(x[np.nonzero(x)], **kwargs)
        if codebook is None:    return None, mse    # No Quantization
        
        # Now we insert 0 as the first item in the codebook:
        newCodebook = np.float32([0]+codebook.tolist())
        memcounts = np.int32([self.numPruned]+memcounts.tolist())
        indexes = (indexes+1).tolist()
        
        flatMask = self.bitMask.flatten().tolist()
        newIndexes = []
        i = 0
        for bit in flatMask:
            if bit == 0:
                newIndexes += [0]
            else:
                newIndexes += [ indexes[i] ]
                i += 1
        newIndexes = np.uint16(newIndexes).reshape(self.shape)
        
        quantizedNetParam = NetParam('NP', newIndexes, newCodebook, self.bitMask, self.trainable,
                                     None if self.name[:8] == "NetParam" else self.name)
        return quantizedNetParam, mse

    # ******************************************************************************************************************
    def getCoreMlWeight(self, how='', dequant=False):
        # Works only for np
        # Note: For fully connected network CoreML needs the transpose of the weight matrix.
        coreMlParam = self.rawVal if self.bitMask is None else (self.rawVal * self.bitMask)
        if how=='T':
            if self.dim==2:
                # Fully Connected Layers -> Simply transpose the 2D matrix
                coreMlParam = coreMlParam.T
            else:
                # Depth-wise Convolution Layers -> Reshape the 4D matrix
                newShape = self.shape
                assert  newShape[3]==1  # Last Dim must be 1
                newShape = (newShape[0], newShape[1], newShape[3], newShape[2])
                coreMlParam = coreMlParam.reshape(newShape)

        if self.codebook is None:   return coreMlParam
        if dequant:
            coreMlParam = self.codebook[coreMlParam]
            if self.bitMask is not None: coreMlParam *= self.bitMask
            return coreMlParam
        
        # For quantized parameters, we need to return a "bytes" stream of all indexes:
        codebookSize = len(self.codebook)
        assert (codebookSize&(codebookSize-1))==0, "Codebook size (=%d) must be a power of 2!"%(codebookSize)
        assert len(self.codebook)<=256, "Codebook size(=%d) must be less than or equal to 256!"%(codebookSize)

        qBits = {256:8, 128:7, 64:6, 32:5, 16:4, 8:3, 4:2, 2:1}[len(self.codebook)]
        masks = [None, 1, 3, 7, 15, 31, 63, 127, 255]
        indexList = coreMlParam.flatten().tolist()
        weightBytes = []
        remainingBits = 8
        curByte = 0
        for i in indexList:
            if remainingBits>=qBits:
                curByte = (curByte<<qBits) + i
                remainingBits -= qBits
                if remainingBits==0:
                    weightBytes += [ curByte ]
                    remainingBits = 8
                    curByte = 0
            else:
                curByte = (curByte<<remainingBits) + (i>>(qBits-remainingBits))
                weightBytes += [ curByte ]
                curByte = i & masks[qBits-remainingBits]
                remainingBits = 8 - qBits + remainingBits

        if remainingBits != 8:
            weightBytes += [ curByte<<remainingBits ]

        return bytes(weightBytes)

    # ******************************************************************************************************************
    def getCoreMlQuantInfo(self):
        # Works only for np
        if self.codebook is None:   return {}
        qBits = {256:8, 128:7, 64:6, 32:5, 16:4, 8:3, 4:2, 2:1}[len(self.codebook)]
        return {"quantization_type":'lut', "nbits":qBits, "quant_lut": self.codebook, "is_quantized_weight":True}

