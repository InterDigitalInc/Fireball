# Copyright (c) 2018-2020 InterDigital AI Research Lab
"""
This file contains the implementation for fireball's print and logging functionality.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 08/22/2020    Shahab                  Started documenting the version history.
# **********************************************************************************************************************
import os
import sys
import datetime

# **********************************************************************************************************************
includeTimeInLogs = False
globalLogger = None
quiet = False

# **********************************************************************************************************************
def isFile(f):
    return isinstance(f,file) if sys.version_info[0] == 2 else hasattr(f, 'read')

# **********************************************************************************************************************
def setQuiet():
    global quiet
    quiet = True

# **********************************************************************************************************************
def setGlobalLogger(logger):
    global globalLogger
    globalLogger = logger

# **********************************************************************************************************************
def includeTime(doInclude=True):
    global includeTimeInLogs
    includeTimeInLogs = doInclude

# **********************************************************************************************************************
def ts():
    global includeTimeInLogs
    if includeTimeInLogs==False:    return ''
    now = datetime.datetime.now()
    return '%02d/%02d/%04d %02d:%02d:%02d '%(now.month, now.day, now.year,
                                             now.hour, now.minute, now.second)

# **********************************************************************************************************************
def interactivePrint(textStr, eol=True, color=None, bold=False, underLine=False):
    if quiet:                       return      # Quiet Mode
    if globalLogger is not None:    return      # Not interactive
    myPrint(textStr, eol, color, bold, underLine)

# **********************************************************************************************************************
def myPrint(textStr, eol=True, color=None, bold=False, underLine=False, getStr=False):
    endCode = ''
    colorCodes = {
                    'red': '\033[91m',
                    'green': '\033[92m',
                    'yellow': '\033[93m',
                    'blue': '\033[94m',
                    'magenta': '\033[95m',
                    'cyan': '\033[96m'
                 }

    if color != None:
        textStr = colorCodes.get(color, '') + textStr
        endCode = '\033[0m'

    if bold:
        textStr = '\033[1m' + textStr
        endCode = '\033[0m'

    if underLine:
        textStr = '\033[4m' + textStr
        endCode = '\033[0m'

    textStr += endCode
    if getStr:  return textStr # getStr always returns without eol regardless of eol param.

    if eol == False:
        sys.stdout.write(textStr)
        sys.stdout.flush()
    else:
        sys.stdout.write(textStr+'\n')
        sys.stdout.flush()

# **********************************************************************************************************************
def printInfo(textStr, logger=None):
    if logger is None: logger = globalLogger
    if isFile(logger):
        # Logger is a File
        if textStr[0]=='\n':    logger.write('\n%s\n'%( myPrint(' %sINFO: %s'%(ts(),textStr[1:]), color='blue', getStr=True) ))
        else:                   logger.write('%s\n'%( myPrint(' %sINFO: %s'%(ts(),textStr), color='blue', getStr=True) ))
        logger.flush()
    elif logger!=None:
        logger.info(textStr.strip('\n'))
    elif textStr[0]=='\n':
        interactivePrint('\n%sINFO: %s'%(ts(),textStr[1:]), color='blue')
    else:
        interactivePrint('%sINFO: %s'%(ts(),textStr), color='blue')

# **********************************************************************************************************************
def printCommand(textStr, logger=None):
    if isFile(logger):
        # Logger is a File
        logger.write( myPrint(textStr, color='cyan', getStr=True) + '\n' )
        logger.flush()
    elif logger!=None:
        logger.info(textStr.strip('\n'))
    else:
        interactivePrint(textStr, color='cyan')

# **********************************************************************************************************************
def printError(textStr, logger=None):
    if isFile(logger):
        # Logger is a File
        if textStr[0]=='\n':    logger.write('\n%s\n'%( myPrint(' %sERROR: %s'%(ts(),textStr[1:]), color='red', getStr=True) ))
        else:                   logger.write('%s\n'%( myPrint(' %sERROR: %s'%(ts(),textStr), color='red', getStr=True) ))
        logger.flush()
    elif logger!=None:
        logger.error(textStr.strip('\n'))
    elif textStr[0]=='\n':
        interactivePrint('\n%sERROR: %s'%(ts(),textStr[1:]), color='red')
    else:
        interactivePrint('%sERROR: %s'%(ts(),textStr), color='red')

# **********************************************************************************************************************
def printWarning(textStr, logger=None):
    if isFile(logger):
        # Logger is a File
        if textStr[0]=='\n':    logger.write('\n%s\n'%( myPrint(' %sWARNING: %s'%(ts(),textStr[1:]), color='yellow', getStr=True) ))
        else:                   logger.write('%s\n'%( myPrint(' %sWARNING: %s'%(ts(),textStr), color='yellow', getStr=True) ))
        logger.flush()
    elif logger!=None:
        logger.warning(textStr.strip('\n'))
    elif textStr[0]=='\n':
        interactivePrint('\n%sWARNING: %s'%(ts(),textStr[1:]), color='yellow')
    else:
        interactivePrint('%sWARNING: %s'%(ts(),textStr), color='yellow')

# **********************************************************************************************************************
# Test Code.
# **********************************************************************************************************************
if __name__ == '__main__':
    includeTime(True)
    printInfo('This is an Info Message!')
    printError('This is an Error Message!')
    printWarning('This is a Warning Message!')
    printInfo('\nThis is an Info Message with new line!')
    printError('\nThis is an Error Message with new line!')
    printWarning('\nThis is a Warning Message with new line!')
    myPrint('RED', color='red')
    myPrint('green', color='green')
    myPrint('yellow', color='yellow')
    myPrint('blue', color='blue')
    myPrint('magenta', color='magenta')
    myPrint('cyan', color='cyan')
    myPrint('bold', bold=True)
    myPrint('underLine', underLine=True)
    myPrint('bold underLine', bold=True, underLine=True)
    myPrint('bold underLine red', color='red', bold=True, underLine=True)
    
    exit(0);

