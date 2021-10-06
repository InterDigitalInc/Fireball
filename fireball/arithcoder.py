# Copyright (c) 2019-2021 InterDigital AI Research Lab
"""
This is an implementation of arithmetic coding used to compress Fireball models.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 02/01/2021    Shahab Hamidi-Rad       Created the file using the latest code from InterDigital's MPEG NNR
#                                       contributions.
# **********************************************************************************************************************
import numpy as np
from .netparam import NetParam

# **********************************************************************************************************************
LOW = np.uint64(0)
HIGH = np.uint64(0xFFFFFFFF)
HALF = (HIGH+1)/2
QRTR = HALF/2
QRTR3 = HIGH+1-QRTR

# **********************************************************************************************************************
class BitStream:
    # ******************************************************************************************************************
    def __init__(self, bytes=None):
        """
            bytes: pass None for encoding case and the actual array of bytes for decoding.
        """
        if bytes is None:
            self.bytes = bytearray([])
        elif type(bytes)==list:
            self.bytes = bytearray(bytes)
        elif type(bytes)==bytearray:
            self.bytes = bytes
        elif type(bytes)==np.ndarray:
            assert bytes.dtype==np.uint8, "Numpy array must be of type uint8."
            self.bytes = bytearray(bytes.tobytes())
        else:
            assert False, "'bytes' must be 'None', a list of integers, a 'bytearray', or a numpy array of type uint8."
            
        self.curByte = 0 if bytes is None else self.bytes[0]
        self.bytePos = 0
        self.bitsAvailable = 8  # always in [1 .. 8]
    
    # ******************************************************************************************************************
    def appendBits(self, newBit, numPending=0):
        """
            Used to append new bits to the bitstream. Used only for encoding. It append one bit equal to "newBit"
            followed by "numPending" bits equal to "~newBit".
        """
        numBits = 1 + numPending
        if newBit == 0: newValue = (1<<numPending)-1
        else:           newValue = (1<<numPending)

        while numBits>0:
            if numBits < self.bitsAvailable:
                self.curByte += newValue << (self.bitsAvailable-numBits)
                self.bitsAvailable -= numBits
                numBits = 0
            else:
                self.curByte += newValue >> (numBits-self.bitsAvailable)
                self.bytes.extend( [self.curByte] )
                numBits -= self.bitsAvailable
                newValue &= (1<<numBits)-1
                self.bitsAvailable = 8
                self.curByte = 0

    # ******************************************************************************************************************
    @property
    def encodedBytes(self):
        """
            Returns the actual encoded bytes. If the last byte is partially filled in, "0"s are appended.
        """
        if self.bitsAvailable == 8:    return self.bytes
        return self.bytes + bytearray([self.curByte])

    # ******************************************************************************************************************
    @property
    def encodedUint8s(self):
        return np.uint8(list(self.encodedBytes))

    # ******************************************************************************************************************
    def numEncodedBits(self):
        """
            Returns the number of encoded bits. Used only for Encoding case.
        """
        return len(self.bytes)*8 + 8-self.bitsAvailable

    # ******************************************************************************************************************
    def numEncodedBytes(self):
        """
            Returns the number of encoded bytes including the last partial byte. Used only for Encoding case.
        """
        if self.bitsAvailable == 8:    return len(self.bytes)
        return len(self.bytes)+1

    # ******************************************************************************************************************
    def reset(self):
        """
            Resets the pointers. Used only for decoding case when you want to rewind and start over.
        """
        self.curByte = self.bytes[0]
        self.bytePos = 0
        self.bitsAvailable = 8  # always in [1 .. 8]

    # ******************************************************************************************************************
    def remainingBits(self):
        """
            Returns the number of remaining bits to decode. Used only for Decoding case.
        """
        return (len(self.bytes)-self.bytePos-1)*8 + self.bitsAvailable
    
    # ******************************************************************************************************************
    def getFirst32Bits(self):
        """
            Returns the first 32 bits of the bitstream. This is only called at the beginning of the decoding process.
        """
        assert (self.bytePos == 0) and (self.bitsAvailable == 8 )
        
        bits = np.uint32(self.bytes[0])
        l = len(self.bytes)
        for i in range(1,4):
            bits <<= 8
            if i<l:
                bits += self.bytes[i]
                self.bytePos = i

        self.bytePos += 1
        if self.bytePos<len(self.bytes):
            self.curByte = self.bytes[self.bytePos]
        return bits
    
    # ******************************************************************************************************************
    def getNextBit( self ):
        """
            Returns the next bit in the bitstream. Used only during the decoding process.
        """
        bit = self.curByte>>(self.bitsAvailable-1) & 1
        self.bitsAvailable -= 1
        if self.bitsAvailable == 0:
            self.bytePos += 1
            self.bitsAvailable = 8
            if self.bytePos<len(self.bytes):
                self.curByte = self.bytes[self.bytePos]
        
        return bit
        
    # ******************************************************************************************************************
    def __repr__(self):
        """
            Returns text string representation of the bitstream for debugging purposes.
        """
        retStr = ''
        for b in self.bytes:
            retStr += format(b, '08b')
        if self.bitsAvailable<8:
            retStr += format((self.curByte>>self.bitsAvailable), '0%db'%(8-self.bitsAvailable))
        return retStr

    # ******************************************************************************************************************
    @classmethod
    def test(cls):
        """
            Unit Test Code.
        """
        print('\nTesting BitStream Class:')
        bitStream = BitStream()
        bitStream.appendBits(1,4)
        bitStream.appendBits(0,3)
        bitStream.appendBits(1,2)
        bitStream.appendBits(0,1)
        if str(bitStream) != '10000011110001':  print('    Testing BitStream.appendBits():      Failed!!!!!!')
        else:                                   print('    Testing BitStream.appendBits():      Passed')

        bytes = bitStream.getEncodedBytes()
        if len(bytes)==2 and bytes[0]==131 and bytes[1]==196:
            print('    Testing BitStream.getEncodedBytes(): Passed')
        else:
            print('    Testing BitStream.getEncodedBytes(): Failed!!!!!!')
                    
        if bitStream.numEncodedBits()==14:  print('    Testing BitStream.numEncodedBits():  Passed')
        else:                               print('    Testing BitStream.numEncodedBits():  Failed!!!!!!')

        if bitStream.numEncodedBytes()==2:  print('    Testing BitStream.numEncodedBytes(): Passed')
        else:                               print('    Testing BitStream.numEncodedBytes(): Failed!!!!!!')

        dataBytes = np.arange(10)
        bitStream2 = BitStream(dataBytes)
        remainingBits1 = bitStream2.remainingBits()

        first32 = bitStream2.getFirst32Bits()
        if first32 == 66051:    print('    Testing BitStream.getFirst32Bits():  Passed')
        else:                   print('    Testing BitStream.getFirst32Bits():  Failed!!!!!!')

        remainingBits2 = bitStream2.remainingBits()
        nextBits = bitStream2.getNextBit()
        remainingBits3 = bitStream2.remainingBits()

        if remainingBits1==80 and remainingBits2==48 and remainingBits3==47:
            print('    Testing BitStream.remainingBits():   Passed')
        else:
            print('    Testing BitStream.remainingBits():   Failed!!!!!!')

        for _ in range(12): bitStream2.getNextBit()
        nextBits1 = bitStream2.getNextBit()
        nextBits2 = bitStream2.getNextBit()
        nextBits3 = bitStream2.getNextBit()
        if nextBits1==1 and nextBits2==0 and nextBits3==1:
            print('    Testing BitStream.getFirst32Bits():  Passed')
        else:
            print('    Testing BitStream.getFirst32Bits():  Failed!!!!!!')

# **********************************************************************************************************************
def encode(rawData, symCounts=None, conditionalProb=True):
    """
        Encodes the bytes in the "rawData" using Arithmetic Coding.
        If the "symCounts" is None, it uses adaptive arithmetic coding. In this case each item is considered a byte and
        the encoder keeps track of the counts for each byte value.
        If the "symCounts" is given, it will be used as a representation of probability distribution of each possible
        value (symbol) in the "rawData".
        The "conditionalProb" is used only if "symCounts" is not None. If it is set to "True", the function updates the
        probability distribution after each symbol is encoded.
    """
    if symCounts is None:
        # Use Adaptive when symCounts is not given. Number of symbols is 256 in this case (plus one for eof).
        conditionalProb = False
        cumCounts = np.uint32(range(258))
        eofSym = 256
    else:
        cumCounts = [0]
        for c in symCounts: cumCounts += [ cumCounts[-1]+c ]
        cumCounts += [ cumCounts[-1]+1 ]    # This is for eof
        cumCounts = np.uint32(cumCounts)
        eofSym = len(cumCounts)-2

    low = LOW
    high = HIGH
    bitStream = BitStream()
    pendingBits = 0
    numUnits = len(rawData)
    for i in range(numUnits+1):
        dataUnit = eofSym if i==numUnits else np.int32(rawData[i])
        curRange = high - low + 1

        plow = cumCounts[dataUnit]
        phigh = cumCounts[dataUnit+1]
        total = cumCounts[-1]
        if symCounts is None:   cumCounts[dataUnit+1:] += 1     # Adaptive
        elif conditionalProb:   cumCounts[dataUnit+1:] -= 1     # Conditional

        high = low + ((curRange*phigh)//total) - 1
        low = low + ((curRange*plow)//total)

        while True:
            if high < HALF:
                bitStream.appendBits(0, pendingBits)
                pendingBits = 0
            elif low >= HALF:
                bitStream.appendBits(1, pendingBits)
                pendingBits = 0
            elif low >= QRTR and high < QRTR3:
                pendingBits += 1
                low -= QRTR
                high -= QRTR
            else:
                break
            low = np.uint64(low*2) & HIGH
            high = np.uint64((high*2)+1) & HIGH
    
    pendingBits += 1
    if low < QRTR:  bitStream.appendBits(0, pendingBits)
    else:           bitStream.appendBits(1, pendingBits)

    return bitStream.encodedUint8s

# **********************************************************************************************************************
def decode(bitStream, symCounts=None, conditionalProb=True):
    """
        Decodes the "bitstream" to an array of original symbols.
        If the "symCounts" is None, it uses adaptive arithmetic coding. In this case each item is considered a byte and
        the decoder keeps track of the counts for each byte value.
        If the "symCounts" is given, it will be used as a representation of probability distribution of each possible
        value (symbol). The "conditionalProb" is used only if "symCounts" is not None. If it is set to "True", the
        function updates the probability distribution after each symbol is decoded.
    """
    if bitStream.__class__.__name__ != 'BitStream':
        bitStream = BitStream(bitStream)
    
    if symCounts is None:
        # Use Adaptive when symCounts is not given. Number of symbols is 256 in this case (plus one for eof).
        conditionalProb = False
        cumCounts = np.arange(258)
        eofSym = 256
    else:
        cumCounts = [0]
        for c in symCounts: cumCounts += [ cumCounts[-1]+c ]
        cumCounts += [ cumCounts[-1]+1 ]    # This is for eof
        cumCounts = np.array(cumCounts)
        eofSym = len(cumCounts)-2

    low = LOW
    high = HIGH
    value = bitStream.getFirst32Bits()
    output = []
    while True:
        curRange = high - low + 1
        scaled = ((value - low + 1) * cumCounts[-1] - 1 ) / curRange

        # Using binary search:
        l = 0
        h = len(cumCounts)-1
        while (h-l)>1:
            m = (l+h)//2
            if scaled<cumCounts[m]: h = m
            else:                   l = m
        
        code = l
        
        plow = cumCounts[l]
        phigh = cumCounts[h]
        total = cumCounts[-1]
        if symCounts is None:   cumCounts[h:] += 1  # Adaptive
        elif conditionalProb:   cumCounts[h:] -= 1  # Conditional
        
        if code==eofSym:  break
            
        output += [code]
        
        high = low + ((curRange*phigh)//total) - 1
        low = low + ((curRange*plow)//total)
        
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QRTR and high < QRTR3:
                value -= QRTR
                low -= QRTR
                high -= QRTR
            else:
                break

            low = np.uint64(low*2) & HIGH
            high = np.uint64((high*2)+1) & HIGH
            value *= 2
            if bitStream.remainingBits()>0:
                value += bitStream.getNextBit()

    if symCounts is None:   return bytearray(output)   # all bytes in adaptive case
    return np.uint32(output)

# **********************************************************************************************************************
# uint, int serialization utility functions
# **********************************************************************************************************************
def uint2ByteList(val):
    # range: 0..536870911 (=2^29 - 1)
    assert val<536870912, "uint2ByteList function can only handle values between 0 and 536,870,911"
    byteList = [ val&0x7F ]
    if val>=128:                    # 2^7
        byteList[0] += 128
        byteList += [ (val>>7)&0x7F ]
    if val>=16384:                  # 2^14
        byteList[1] += 128
        byteList += [ (val>>14)&0x7F ]
    if val>=2097152:                # 2^21
        byteList[2] += 128
        byteList += [ (val>>21)&0xFF ]
    return bytearray(byteList)

# **********************************************************************************************************************
def byteList2Uint(byteList, offset=None):
    dataBytes = byteList if offset is None else byteList[offset:]
    uintLen = 4
    uintValue = dataBytes[0]&0x7F
    if dataBytes[0]<128:            uintLen = 1
    else:
        uintValue += (np.uint32(dataBytes[1]&0x7F)<<7)
        if dataBytes[1]<128:        uintLen = 2
        else:
            uintValue += (np.uint32(dataBytes[2]&0x7F)<<14)
            if dataBytes[2]<128:    uintLen = 3
            else:
                uintValue += (np.uint32(dataBytes[3])<<21)

    if offset is None:  return np.uint32(uintValue)
    return np.uint32(uintValue), (offset+uintLen)

# **********************************************************************************************************************
def int2ByteList(val):
    # range: -268435455..268435455 (= +/- 2^28 - 1)
    isNeg = val<0
    if isNeg:   val = -val
    assert val<268435456, "int2ByteList function can only handle values between -268,435,455 and 268,435,456"

    byteList = [ val&0x3F ]
    if isNeg:   byteList[0] += 128
    if val>=64:                    # 2^6
        byteList[0] += 64
        byteList += [ (val>>6)&0x7F ]
    if val>=8192:                  # 2^13
        byteList[1] += 128
        byteList += [ (val>>13)&0x7F ]
    if val>=1048576:                # 2^20
        byteList[2] += 128
        byteList += [ (val>>20)&0xFF ]
    return bytearray(byteList)

# **********************************************************************************************************************
def byteList2Int(byteList, offset=None):
    dataBytes = byteList if offset is None else byteList[offset:]
    intLen = 4
    intValue = dataBytes[0]&0x3F
    if (dataBytes[0]&0x7F)<64:      intLen = 1
    else:
        intValue += (np.uint32(dataBytes[1]&0x7F)<<6)
        if dataBytes[1]<128:        intLen = 2
        else:
            intValue += (np.uint32(dataBytes[2]&0x7F)<<13)
            if dataBytes[2]<128:    intLen = 3
            else:
                intValue += (np.uint32(dataBytes[3])<<20)

    if dataBytes[0]>=128:   intValue = -intValue  # Apply Signbit
    if offset is None:  return np.int32(intValue)
    return np.int32(intValue), (offset+intLen)

# **********************************************************************************************************************
def str2ByteList(strVal):
    strBytes = strVal.encode('utf-8')
    byteList = uint2ByteList(len(strBytes))
    byteList += strBytes
    return byteList
    
# **********************************************************************************************************************
def byteList2Str(byteList, offset=None):
    dataBytes = byteList if offset is None else byteList[offset:]
    strLen,b  = byteList2Uint(dataBytes, 0)
    strVal = dataBytes[b:strLen+b].decode('utf-8')
    
    if offset is None: return strVal
    return strVal, (offset+strLen+b)

# **********************************************************************************************************************
def encodeNetParam(netParam):
    netParamDic = {}
    def putUintList(uintList):
        data = uint2ByteList(len(uintList))
        for u in uintList:    data += uint2ByteList(u)
        return data
    
    def putSymCounts(indexes, numSym):
        symCountIdx, symCountVals = np.unique(indexes, return_counts=True)
        symCounts = np.uint32([0]*numSym)
        symCounts[symCountIdx]=symCountVals
        return putUintList(symCounts), symCounts
    
    if netParam.codebook is None:
        if netParam.bitMask is None:
            # Not Quantized - Not Pruned
            # The adaptively encoded data includes: shape, flags, and the actual floating point values
            data = putUintList(netParam.shape)
            data += uint2ByteList(1 if netParam.trainable else 0)
            data += netParam.rawVal.tobytes()
            return {'r0': encode(data) }
            
        # Not Quantized - Pruned:
        # Get a list of uint32s containing packed bitMasks (mask32s) and a list of float32s containing
        # the non-zero values (nzVals)
        nzVals, mask32s = netParam.packMasks()
        mask32sBytes = mask32s.tobytes()
        nzValsBytes = nzVals.tobytes()
        
        # The adaptively encoded data includes: shape, flags, No. of bytes in packed bitMasks, packed bitMasks,
        # and non-zero values
        data = putUintList(netParam.shape)
        data += uint2ByteList(1 if netParam.trainable else 0)
        data += uint2ByteList(len(mask32sBytes))
        data += mask32sBytes + nzValsBytes
        return {'p0': encode(data) }

    if netParam.bitMask is None:
        # Quantized - Not pruned
        # The adaptively encoded data includes: shape, flags, symbol counts, and the codebook floating point values
        # The indexes are conditionally encoded.
        flatIndexes = netParam.rawVal.flatten()
        symCountBytes, symCounts = putSymCounts(flatIndexes, netParam.codebookSize)
        
        data = putUintList(netParam.shape)
        data += uint2ByteList(1 if netParam.trainable else 0)
        data += symCountBytes + netParam.codebook.tobytes()
        return {'q0': encode(data),
                'i0': encode(flatIndexes, symCounts) }
        
    # Quantized - pruned
        
    # Get a list of uint32s containing packed bitMasks (mask32s) and a list of non-zero indexes (nzIndexes)
    nzIndexes, mask32s = netParam.packMasks()
    symCountBytes, symCounts = putSymCounts(nzIndexes-1, netParam.codebookSize-1)
    mask32sBytes = mask32s.tobytes()
    codebookBytes = netParam.codebook[1:].tobytes()     # Codebook info (not including the first item for 0)

    # The adaptively encoded data includes: shape, flags, Num packed bitMasks bytes, packed bitMasks bytes,
    # symbol count bytes, and the codebook bytes
    # The non-zero indexes (minus 1) are conditionally encoded.
    data = putUintList(netParam.shape)
    data += uint2ByteList(1 if netParam.trainable else 0)
    data += uint2ByteList( len(mask32sBytes) )
    data += mask32sBytes + symCountBytes + codebookBytes
    return {'p0': encode(data), 'i0': encode(nzIndexes-1, symCounts) }

# **********************************************************************************************************************
def decodeNetParam(netParamDic):
    def getUintList(data):
        n, b = byteList2Uint(data, 0)
        uintList = []
        for _ in range(n):
            u, b = byteList2Uint(data, b)
            uintList += [ u ]
        return uintList, b
            
    if 'r0' in netParamDic:
        # Not Quantized - Not Pruned
        # The adaptively encoded data includes: shape and the actual floating point values
        data = decode(netParamDic['r0'])
        shape, b = getUintList(data)
        flags, b = byteList2Uint(data, b)
        rawVal = np.frombuffer(data[b:], np.float32).reshape(shape)
        return NetParam('NP', rawVal, None, None, (flags&1)==1)

    if 'p0' in netParamDic:
        if 'i0' in netParamDic:
            # Quantized - pruned
            # The adaptively encoded data includes: sparseInfoLen, sparseInfoBytes, codebookBytes
            # The non-zero indexes are conditionally encoded.
            data = decode(netParamDic['p0'])
            shape, b = getUintList(data)
            flags, b = byteList2Uint(data, b)

            mask32Len, b = byteList2Uint(data, b)
            mask32Bytes = data[b : b+mask32Len]
            symCounts, symCountsLen = getUintList( data[b+mask32Len:] )
            codebookBytes = data[b+mask32Len+symCountsLen :]
            mask32 = np.frombuffer(mask32Bytes, dtype=np.uint32)
            codebook = np.frombuffer(codebookBytes, dtype=np.float32)
            codebook = np.float32([0] + codebook.tolist())  # Add the '0' entry at the beginning of codebook
            
            nzFlatIndexes = decode(netParamDic['i0'], symCounts) + 1
            indexes, bitMask = NetParam.unpackMasks(nzFlatIndexes, mask32, shape)
            return NetParam('NP', indexes, codebook, bitMask, (flags&1)==1)

        # Not Quantized - Pruned
        # The adaptively encoded data includes: sparseInfoLen, sparseInfoBytes, nzValsBytes
        data = decode(netParamDic['p0'])
        shape, b = getUintList(data)
        flags, b = byteList2Uint(data, b)
        mask32Len, b = byteList2Uint(data, b)
        mask32Bytes = data[b : b+mask32Len]
        nzValsBytes = data[b+mask32Len :]
        mask32 = np.frombuffer(mask32Bytes, dtype=np.uint32)
        nzVals = np.frombuffer(nzValsBytes, dtype=np.float32)
        paramVals, bitMask = NetParam.unpackMasks(nzVals, mask32, shape)
        return NetParam('NP', paramVals, None, bitMask, (flags&1)==1)
        
    # Quantized - Not pruned
    # The adaptively encoded data includes: dimensions, shape, and the codebook floating point values
    # The indexes are conditionally encoded.
    assert 'q0' in netParamDic
    assert 'i0' in netParamDic
    data = decode(netParamDic['q0'])
    shape, b = getUintList(data)
    flags, b = byteList2Uint(data, b)
    symCounts, symCountsLen = getUintList( data[b:] )
    codebookBytes = data[b+symCountsLen :]
    codebook = np.frombuffer(codebookBytes, dtype=np.float32)
    indexes = decode( netParamDic['i0'], symCounts ).reshape(shape)
    return NetParam('NP', indexes, codebook, None, (flags&1)==1)
    
