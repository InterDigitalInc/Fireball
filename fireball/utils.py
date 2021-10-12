# Copyright (c) 2017-2020 InterDigital AI Research Lab
"""
This file contains some common utility functions that are used by the modules in Fireball library.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 07/24/2019    Shahab                  Created the file by gathering the functions from
#                                       different files.
# **********************************************************************************************************************
import numpy as np
import os

# **********************************************************************************************************************
# python 2/3 compatibility (Remove when dropping support for python2.7)
try:
    # Python 2
    range = xrange
    def keyVals(x): return x.iteritems()
    PYTHON_STR = "python2"
except NameError:
    # Python 3
    PYTHON_STR = "python3"
    def keyVals(x): return x.items()
    unicode = str

# **********************************************************************************************************************
def floatStr(f, strLen):
    """
    floatStr
    A utility function used to convert a floating-point number to a simplified string.
    
    Arguments:
        f: Required, float
            A floating-point number to be converted to the simplified text string.
        strLen: Required, int
            The exact length of the text string returned.
            
    Returns: string
        A text string of length "strLen" containing the floating-point number.
    """
    assert strLen>3
    remainingLen = strLen
    returnStr = ''
    if f<0:
        returnStr = '-'
        remainingLen -= 1
        f = -f

    intStr, fracStr = (('%%.%df'%(15))%f).split('.')

    if f==0:
        returnStr = "0"
        
    elif intStr == '0':
        fracStr = (('%%.%df'%(remainingLen - 2))%f).split('.')[1].rstrip('0')
        if fracStr == '':
            if strLen<6:  returnStr += "0.0"[:remainingLen]
            else:
                returnStr += "%0.*e"%(remainingLen-6,f)
                f,p = returnStr.split('e')
                returnStr = f.rstrip('0').rstrip('.') + 'e' + p
        else:
            returnStr += '0.'+fracStr

    elif len(intStr)==remainingLen:
        returnStr += intStr
    elif len(intStr)<remainingLen:
        returnStr += (intStr + '.' + fracStr)[:remainingLen].rstrip('0')
        if fracStr.rstrip('0') == '': returnStr = returnStr.rstrip('.')
    else:
        if strLen>=5:
            returnStr += "%0.*e"%(remainingLen-6,f)
            f,p = returnStr.split('e')
            returnStr = f.rstrip('0').rstrip('.') + 'e' + p
        else:
            returnStr += strLen*'#'

    if len(returnStr)<strLen:  returnStr += (strLen-len(returnStr))*' '
    return returnStr

# **********************************************************************************************************************
def getCurrentGpus(returnHalfGpus=False):
    """
    getCurrentGpus
    This is function gets list of GPU devices from TensorFlow library and returns an integer array of GPU numbers
    for the detected GPU devices.
        
    Arguments:
        returnHalfGpus: Boolean, default:False
            If true, Only half of the detected devices will be returned in the integer array.
            Otherwise the GPU numbers of all detected GPUs will be returned.
    
    Returns:
        If an integer array of GPU numbers for the detected GPU devices. If no GPU device was found or an error happens
        while detecting the GPU devices, this function returns a list with a single item of -1 in it.
    """
    try:
        from tensorflow.python.client import device_lib
        localDeviceProtos = device_lib.list_local_devices()
        gpus = []
        for x in localDeviceProtos:
            if x.device_type.lower() != 'gpu': continue
            gpus += [ int(x.name.split(':')[-1]) ]
        if returnHalfGpus:
            if len(gpus)>=2:
                return gpus[:len(gpus)//2]
        if len(gpus)==0:
            return [-1]
    except:
        print('ERROR: Cannot get list of current GPUs! Using only CPU.')
        return [-1]

    return gpus
