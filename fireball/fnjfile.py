# Copyright (c) 2020 InterDigital AI Research Lab
"""
Implementation of the FNJ file format. Starting version 1.3, Fireball uses a new
file format called FNJ (Fireball Numpy JSON).
The main reason of this migration was the incompatibility issues with numpy files
(NPZ) across different numpy versions and its dependability on pickle package.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 08/12/2020    Shahab Hamidi-Rad       First version of the file.
# 08/18/2020    Shahab                  Added support for squeezing int tensors (for quantization). This is backward
#                                       compatible. Due to slow file save/load when in this mode, this feature is
#                                       currently disabled by default.
# **********************************************************************************************************************
import numpy as np
import json

import fireball.arithcoder as ac

# **********************************************************************************************************************
def intsToBits(ints, bitsPerInt):
    availableBits = 0  # remaining unused bits in current byte
    availableWord = 0
    bits = []
    i=0
    while True:
        while availableBits >= 8:
            availableBits -= 8
            bits += [ (availableWord >> availableBits) & 255 ]

        if i>=len(ints): break
        availableWord = (availableWord<<bitsPerInt) + ints[i]
        availableBits += bitsPerInt
        i += 1

    assert availableBits < 8
    if availableBits>0:
        bits += [ availableWord << (8-availableBits)  ]
    
    return bytes(bits)

# **********************************************************************************************************************
def bitsToInts(bits, bitsPerInt):
    bits = list(bits)
    availableBits = 0  # remaining unused bits in current byte
    availableWord = np.uint16(0)
    ints = []
    b=0
    mask = (1<<bitsPerInt)-1
    while True:
        while availableBits >= bitsPerInt:
            availableBits -= bitsPerInt
            ints += [ (availableWord >> availableBits) & mask ]
        
        if b>=len(bits): break
        availableWord = (availableWord << 8) + int(bits[b])
        availableBits += 8
        b += 1
        
    return np.array(ints, dtype=np.uint8 if bitsPerInt<=8 else np.uint16)

# **********************************************************************************************************************
def getNpBytes(data, npBytes, squeezeUints=False):
    newVal = None
    if type(data) == dict:
        for k, v in data.items():
            value, npBytes = getNpBytes(v, npBytes, squeezeUints)
            if value is not None: data[k]= value
    
    elif type(data) == list:
        for k,v in enumerate(data):
            value, npBytes = getNpBytes(v, npBytes, squeezeUints)
            if value is not None: data[k]= value

    elif type(data) == tuple:
        newVal = []
        for k,v in enumerate(data):
            value, npBytes = getNpBytes(v, npBytes, squeezeUints)
            if value is not None:   newVal += (value,)
            else:                   newVal += (v,)
        newVal = {"$$$TUPLE$$$":newVal}

    elif type(data) == np.ndarray:
        typeVal = -1
        for i, x in enumerate([np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
                               np.float16, np.float32, np.float64]):
            if x == data.dtype:
                typeVal=i+1
                break
        assert typeVal!=-1, "Saving FNJ file: Arrays of type '%s' are not supported!"%(str(data.dtype))
        
        if squeezeUints and (data.dtype in [np.uint8, np.uint16]):     # Squeeze uint8 and uint16 data
            # This makes the quantized files smaller at the cost of more loading time when the model is read back.
            bitsPerInt = int(np.ceil(np.log2(data.max()+1)))
            dataBytes = intsToBits(data.flatten().tolist(), bitsPerInt)
            typeVal = 100 + bitsPerInt     # Mark Data as Squeezed and include bit-depth into the typeVal
        else:
            dataBytes = data.tobytes()
            
        headerBytes = np.array([len(data.shape)] + [len(dataBytes)] + list(data.shape) + [typeVal], dtype=np.int32).tobytes()
        newVal = "npParam@%d-%d"%(len(npBytes), len(headerBytes))
        npBytes += headerBytes
        npBytes += dataBytes
        
    return newVal, npBytes

# **********************************************************************************************************************
def readNpInfo(data, file):
    newVal = None
    if type(data) == dict:
        for k, v in data.items():
            value = readNpInfo(v, file)
            if value is not None: data[k]= value
        
        if len(data)==1 and "$$$TUPLE$$$" in data:
            newVal = tuple(data["$$$TUPLE$$$"])

    elif type(data) == list:
        for k,v in enumerate(data):
            value = readNpInfo(v, file)
            if value is not None: data[k]= value

    elif type(data) == tuple:
        newVal = ()
        for k,v in enumerate(data):
            value = readNpInfo(v, file)
            if value is not None:   newVal += (value,)
            else:                   newVal += (v,)

    elif type(data) == str:
        if data[:8] == "npParam@":
            pos, headerLen = (int(x) for x in data[8:].split('-'))
            file.seek(pos,0)
            headerBytes = file.read(headerLen)
            header = np.frombuffer(headerBytes, dtype=np.int32)
            dataShape = header[2:header[0]+2].tolist()
            dataSize = header[1]
            typeVal = header[-1]
            dataBytes = file.read(dataSize)
            if typeVal > 100:   # Squeezed integer values
                bitsPerInt = typeVal - 100
                numInts = np.prod(dataShape)
                newVal = bitsToInts(dataBytes, bitsPerInt)[:numInts].reshape(dataShape)
            else:
                dtype = [None, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
                               np.float16, np.float32, np.float64][typeVal]
                newVal = np.frombuffer(dataBytes, dtype=dtype).reshape(dataShape)

    return newVal

# **********************************************************************************************************************
def saveFNJ(fileName, dic, squeezeUints=False, fileInfo=None, flags=0):
    version = [1,2]     # 1.02
    file = open(fileName, 'wb')
    savedBytes = bytearray("Fireball-Numpy-Json ", 'utf-8')
    savedBytes += np.array(version,dtype=np.uint8).tobytes()
    flagsJsonPosPos = len(savedBytes)
    flagsJsonPosInfo = np.array([flags,0,0],dtype=np.uint32)
    savedBytes += flagsJsonPosInfo.tobytes()
    
    _, savedBytes = getNpBytes(dic, savedBytes, squeezeUints)
    
    if fileInfo is not None:
        flagsJsonPosInfo[1] = len(savedBytes)
        if (flags & 1): savedBytes += ac.encode( bytes(json.dumps(fileInfo), 'utf-8') ).tobytes()
        else:           savedBytes += bytes(json.dumps(fileInfo), 'utf-8')
    
    flagsJsonPosInfo[2] = len(savedBytes)
    if (flags & 1): savedBytes += ac.encode( bytes(json.dumps(dic), 'utf-8') ).tobytes()
    else:           savedBytes += bytes(json.dumps(dic), 'utf-8')
    flagsJsonPosInfoBytes = flagsJsonPosInfo.tobytes()
    savedBytes[flagsJsonPosPos:flagsJsonPosPos+len(flagsJsonPosInfoBytes)] = flagsJsonPosInfoBytes
    file.write( savedBytes )
    file.close()

# **********************************************************************************************************************
def loadFNJ(fileName):
    file = open(fileName, 'rb')
    x = file.read(20)
    assert(x == bytes("Fireball-Numpy-Json ", 'utf-8')), "Invalid file signature!"

    buf = file.read(2)
    version = np.frombuffer(buf, dtype=np.uint8).tolist()
    version = version[0] + version[1]/100.0
    if version >= 1.02:
        buf = file.read(12)
        flags, fileInfoPos, dataDicPos = np.frombuffer(buf, dtype=np.int32)
    else:
        # Previous versions (before 1.02) did not support flags
        flags = 0
        buf = file.read(8)
        fileInfoPos, dataDicPos = np.frombuffer(buf, dtype=np.int32)
        
    if fileInfoPos>0:
        file.seek(fileInfoPos, 0)
        buf = file.read(dataDicPos-fileInfoPos)
        if (flags & 1): fileInfoDic = json.loads(ac.decode(buf))
        else:           fileInfoDic = json.loads(buf)
    
    file.seek(0,2)
    fileSize = file.tell()
    file.seek(dataDicPos, 0)
    buf = file.read(fileSize-dataDicPos)
    if (flags & 1): dataDic = json.loads(ac.decode(bytearray(buf)))
    else:           dataDic = json.loads(buf)
    readNpInfo(dataDic, file)
    file.close()
    return dataDic

