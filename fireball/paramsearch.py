# Copyright (c) 2018-2020 InterDigital AI Research Lab
"""
This file contains the implementation for fireball's parameter search functionality.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 11/17/2016    Shahab Hamidi-Rad       First version of the file.
# 12/14/2016    Shahab                  Added support for automatic handling of new lines in the description of the
#                                       options.
# 08/22/2020    Shahab                  Added support for specific combinations.
# **********************************************************************************************************************

import numpy as np
import os
import sys
import subprocess
import time
import yaml
import shlex
import threading

from .printutils import *
from .utils import *        # Utility functions

# **********************************************************************************************************************
# NOTE: To kill this program and all sub-processes, use "sudo kill -2 <PID>". This will send Ctrl+C
# which is handled by this program to clean up the sub-processes.

# **********************************************************************************************************************
logFile = None
shouldStop = False  # Set to true when Ctrl+C is pressed
quietProcesses = False
threadStates = None
numWorkers = None
availableGpus = [-1]
gpusForThread = None
availableThreads = None

# **********************************************************************************************************************
def startWorker(threadId, params):
    resultsFileName = '%sParamSearch.csv'%(sys.argv[0].split('.')[0])
    command = '%s %s -saveResults=%s '%(PYTHON_STR, sys.argv[0], resultsFileName)

    if 'workerId' in params:    params['workerId'] = threadId
    for key, value in keyVals(params):
        if value is None:       command += ' -'+key
        elif str(value)=='':    command += ' -'+key
        elif str(value)=='+':   command += ' -'+key
        elif value!='-':        command += ' -%s=%s'%(key, str(value))
        
    if quietProcesses == False:
        printCommand('    ' + command, logFile)
    else:
        printInfo('Worker %02d: Command-Line:\n    %s'%(threadId, command), logFile)

    FNULL = open(os.devnull, 'w') if quietProcesses else None
    process = subprocess.Popen(shlex.split(command), shell=False, stdout=FNULL, stderr=subprocess.STDOUT)
    while process.poll() is None:
        if shouldStop:  process.terminate()
        time.sleep(.5)

    if (process.returncode != 0) and (shouldStop==False):
        printError('Worker %02d: The process returned %d!'%(threadId, process.returncode), logFile)

# **********************************************************************************************************************
def processOneCombination(params, combinationStr, combinationNo, threadId):
    global availableThreads
    threadStates[threadId]['workingOn'] = combinationStr
    startWorker(threadId, params)
    if shouldStop:
        printInfo('Worker %02d: Forced Stop while Processing combination %d (%s).'%(threadId, combinationNo, combinationStr), logFile)
    else:
        updateProgressAndSave(combinationStr, combinationNo, threadId)
    availableThreads += [threadId]

# **********************************************************************************************************************
savingInProgress = False
def updateProgressAndSave(combinationStr, combinationNo, threadId):
    global savingInProgress
    while savingInProgress:
        while savingInProgress: time.sleep(.5)
        time.sleep(threadId*0.1)

    savingInProgress = True
    if (threadId!=0) and (combinationStr is not None):
        threadStates[threadId]['processed'] += [combinationStr]
        threadStates[threadId]['workingOn'] = None
        printInfo('Worker %02d: Finished Processing combination %d (%s).'%(threadId, combinationNo, combinationStr), logFile)

    yamlData = { 'numWorkers': numWorkers, 'threadStates': threadStates }
    yamlFile = '%sParamSearch.yml'%(sys.argv[0].split('.')[0])
    with open(yamlFile, 'w') as outfile:
        outfile.write( yaml.dump(yamlData) )

    savingInProgress = False

# **********************************************************************************************************************
def paramSearch(parameterChoices, additionalParams=None, nWorkers=1, doLog=True, gpus=None, restart=False):
    # parameterChoices: List of tuples of the form: (paramName, [value0, value1, ...])
    #    For boolean values use the '+' value for presence and '-' for non-presence of the option in the command-line.
    if doLog:
        global logFile
        logFile = open('%sParamSearch.log'%(sys.argv[0].split('.')[0]), mode='w')
        includeTime()       # Include time in log messages

    if gpus is not None:
        global availableGpus
        availableGpus = [ int(x) for x in gpus.split(',') ]
    
    global numWorkers
    numWorkers = nWorkers

    if type(parameterChoices) == list:
        # Getting a list of all combinations of parameter choices. (in reverse order)
        remainingCombinations = ['C']
        paramNames = []
        for paramName, choices in parameterChoices:
            paramNames += [paramName]
            numChoices = len(choices)
            remainingCombinations = [ '%s.%d'%(combination,i) for combination in remainingCombinations for i in range(numChoices-1,-1,-1)]
        parameterChoices = dict(parameterChoices)
    elif type(parameterChoices) == dict:
        paramNames = parameterChoices['params']
        choices = parameterChoices['choices']
        remainingCombinations = ['C%d'%(c) for c in range(len(choices),0,-1)]


    global threadStates
    yamlFile = '%sParamSearch.yml'%(sys.argv[0].split('.')[0])
    if restart:
        if os.path.exists( yamlFile ): os.remove( yamlFile )
        resultsFileName = '%sParamSearch.csv'%(sys.argv[0].split('.')[0])
        if os.path.exists( resultsFileName ): os.remove( resultsFileName )

    if os.path.exists( yamlFile ):
        yamlData = yaml.load(open(yamlFile, 'r'))
        numWorkers = yamlData['numWorkers']
        threadStates = yamlData['threadStates']
        printWarning('\nDetected interrupted precess.', logFile)
        printInfo('Skipping the following Combinations:', logFile)
        for threadInfo in threadStates:
            threadInfo['workingOn'] = None
            for processedCombination in threadInfo['processed']:
                printInfo('       ' + processedCombination, logFile)
                remainingCombinations.remove(processedCombination)
    else:
        threadStates = [{'processed': [], 'workingOn':None} for _ in range(numWorkers+1)]

    global quietProcesses
    quietProcesses = (logFile is not None) or (numWorkers>1)

    # Assign GPUs to Workers
    global gpusForThread
    numGpus = len(availableGpus)
    gpusForThread = ['']*numWorkers
    for x in range(max(numWorkers,numGpus)):
        gpusForThread[x%numWorkers] += '%s%d'%('' if gpusForThread[x%numWorkers]=='' else ',',availableGpus[x%numGpus])

    printInfo('Trying %d combinations of parameters using %d worker thread(s) and %d GPU(s)'%(len(remainingCombinations),
                                                                                              numWorkers, numGpus), logFile)
    printInfo('To stop this process use the command: "sudo kill -2 %d"'%(os.getpid()), logFile)

    global availableThreads
    availableThreads = [x for x in range(numWorkers,0,-1)]
    try:
        combinationNo = 0
        while True:
            if len(remainingCombinations) == 0:
                while len(availableThreads) < numWorkers: time.sleep(0.5)
                break   # We are done
            nextCombinationStr = remainingCombinations.pop()

            # Wait for a thread to become available
            while len(availableThreads) == 0:  time.sleep(0.5)
            threadId = availableThreads.pop()

            if '.' in nextCombinationStr:
                combination = nextCombinationStr.split('.')[1:]
                params = { paramName: parameterChoices[paramName][ int(combination[i]) ] for i, paramName in enumerate(paramNames) }
            else:
                choiceNo = int(nextCombinationStr[1:])-1
                params = dict(zip(paramNames, choices[choiceNo]))
                
            params['gpus'] = gpusForThread[threadId-1]
            if additionalParams is not None:
                params.update(additionalParams)

            threading.Thread(target = processOneCombination, args = (params, nextCombinationStr, combinationNo, threadId,)).start()
            combinationNo += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        global shouldStop
        shouldStop = True
        printWarning("\nCtrl+C => Waiting for all worker threads and processes to stop.", logFile)
        while len(availableThreads) < numWorkers: time.sleep(0.5)
        exit(1)

    printInfo('\nDone processing all combination of parameters.', logFile)
    printInfo('Now Cleaning up.', logFile)
    os.remove( yamlFile )
    return True

# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
def saveResults(resultsCsvFile, paramValues, model=None):
    import fcntl
    import errno
    import time
    import os
    def lockFile(theFile):
        while True:
            try:
                # This throws an errno.EAGAIN=35 if file is locked by others
                fcntl.flock(theFile, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(0.1) # Making sure the other process is done with the file
                break
            except IOError as e:
                if e.errno != errno.EAGAIN:     raise   # raise any unrelated IOErrors
                else:                           time.sleep(0.1)

    def unlockFile(theFile):
        fcntl.flock(theFile, fcntl.LOCK_UN)

    paramNames, paramValues = zip(*paramValues)
    
    def quotedStr(s):
        return '"%s"'%(str(s)) if ',' in str(s) else str(s)
        
    if os.path.exists(resultsCsvFile)==False:
        header = ','.join(paramNames)
        if model is not None:
            if model.results is not None:
                if 'csvItems' in model.results:
                    resultsHeaderStr = ','.join( quotedStr(x) for x in model.results['csvItems'] )
                    header +=  ',' + resultsHeaderStr
                    
        resultsFile = open(resultsCsvFile, 'w')
        lockFile(resultsFile)
        resultsFile.write(header + '\n')
        unlockFile(resultsFile)
        resultsFile.close()

    rowStr = ','.join( quotedStr(paramValue) for paramValue in paramValues )
    if model is not None:
        if model.results is not None:
            if 'csvItems' in model.results:
                resultsRowStr = ','.join( quotedStr(model.results[x]) for x in model.results['csvItems'] )
                rowStr +=  ',' + resultsRowStr

    resultsFile = open(resultsCsvFile, 'a')
    lockFile(resultsFile)
    resultsFile.write(rowStr + '\n')
    unlockFile(resultsFile)
    resultsFile.close()


