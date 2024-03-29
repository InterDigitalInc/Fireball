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
def printVersionInfo():
    """
    printVersionInfo
    This is function prints version information about Fireball, python, and some of the main packages used by Fireball.
    """
    from . import __version__
    import tensorflow as tf
    import sys
    print("Fireball Version:        %s"%(__version__))
    print("Python Version:          %s"%(sys.version))
    print("TensorFlow Version:      %s"%(tf.__version__))
    print("Numpy Version:           %s"%(np.__version__))
    try:
        import onnx
        print("ONNX Version:            %s"%(onnx.__version__))
    except:
        pass
    try:
        import onnxruntime
        print("ONNX Runtime Version:    %s"%(onnxruntime.__version__))
    except:
        pass
    try:
        import coremltools
        print("CoreML Version:          %s"%(coremltools.__version__))
    except:
        pass
    try:
        import cv2
        print("OpenCV Version:          %s"%(cv2.__version__))
    except:
        pass
    try:
        availGPUs = tf.config.list_physical_devices('GPU')  # Available physical GPUs
        if availGPUs:
            print("GPUs:                    %d"%(len(availGPUs)))
            for i,gpu in enumerate(availGPUs):
                print("  GPU%d:                  %s"%(i, tf.config.experimental.get_device_details(gpu)['device_name']))
        else:
            print("GPUs:                    %s"%("None"))
    except:
        pass
