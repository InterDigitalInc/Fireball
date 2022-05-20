# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the functionality to export current model to TensorFlow. It composes
a python file which contains the model definition and a numpy compressed file (.npz)
containing the parameters of the network.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 10/25/2019    Shahab Hamidi-Rad       Created the file.
# 10/29/2020    Shahab                  Completed the first version for Tensorflow export functionality.
# 11/29/2020    Shahab                  Created the TfBuilder class and moved all the functionality into
#                                       this class.
# 05/12/2022    Shahab                  A Minor fix to support TF2.
# **********************************************************************************************************************
import numpy as np
import tensorflow as tf
import time
import os
import datetime
from .printutils import myPrint
from . import __version__ as fbVersion

# ******************************************************************************************************************
def getStr(indent, lines, params=None):
    if type(lines) == str:  tfStr = indent + lines
    else:                   tfStr = '\n'.join([ (indent + line) for line in lines])
    if params is not None:
        if type(params) == tuple:   tfStr = tfStr % params
        else:                       tfStr = tfStr % (params)
    tfStr += '\n'
    return tfStr

# **********************************************************************************************************************
class TfBuilder:
    def __init__(self, fbModel):
        self.fbModel = fbModel
        self.graphIndent = 0
        self.runQuantized = False
        self.classNames = None
        self.tfFile = None
        self.definedMethods = set()

    # ******************************************************************************************************************
    def getScopeStr(self, scope):
        if scope[0] not in ["'",'"']:   scope = "'%s'"%scope
        return "with tf.name_scope(%s), tf1.variable_scope(%s,reuse=tf1.AUTO_REUSE):"%(scope, scope)

    # ******************************************************************************************************************
    def writeHeader(self, modelFileName):
        now = datetime.datetime.now()
        nowStr = '%02d/%02d/%04d %02d:%02d:%02d'%(now.month, now.day, now.year, now.hour, now.minute, now.second)
        self.tfFile.write( getStr("",
                                  ("#!/usr/bin/env python3",
                                   "# -*- coding: utf-8 -*-",
                                   "#",
                                   "# %s: (Automatically Generated on %s)",
                                   "# This is the TensorFlow code for model \"%s\"",
                                   "# Exported by Fireball version "+fbVersion,
                                   "",
                                   "import numpy as np",
                                   "import os",
                                   "import warnings",
                                   "warnings.simplefilter(action='ignore', category=FutureWarning)",
                                   "import tensorflow as tf",
                                   "try:    import tensorflow.compat.v1 as tf1",
                                   "except: tf1 = tf",
                                   "try:    import tensorflow.random as tfr",
                                   "except: tfr = tf",
                                   "tf1.disable_eager_execution()",
                                   "",
                                   "SEED = 36478",
                                   "exportPath = os.path.dirname(os.path.realpath(__file__))",
                                   "",
                                   "class Network:",
                                   "    def __init__(self, paramsFilePath=None, **kwargs):",
                                   "        tf1.reset_default_graph()",
                                   "        train = kwargs.get('train', False)",
                                   "        self.l2Loss = 0",
                                   ""),
                                  (modelFileName, nowStr, self.fbModel.name) ) )
        
    # ******************************************************************************************************************
    def addToSection(self, section, lines, params=None):
        if section== 'method':      indent = '    '
        elif section == 'graph':    indent = '    '*(self.graphIndent+2)  # inside the 'buildGraph' function
        else:                       indent = '        '  # inside the '__init__', 'infer', or 'trainBatch' functions
        if section == 'init':       self.initStr += getStr(indent, lines, params)
        elif section == 'method':   self.methodsStr += getStr(indent, lines, params)
        elif section == 'graph':    self.graphStr += getStr(indent, lines, params)
        elif section == 'infer':    self.inferStr += getStr(indent, lines, params)
        elif section == 'train':    self.trainStr += getStr(indent, lines, params)
        else:                       raise NotImplementedError("'%s' section not implemented!"%(section))

    # ******************************************************************************************************************
    def addToInit(self, initLines, params=None):    self.addToSection('init', initLines, params)
    def addMethod(self, methodLines, params=None):  self.addToSection('method', methodLines, params)
    def addToGraph(self, graphLines, params=None):  self.addToSection('graph', graphLines, params)
    def addToInfer(self, inferLines, params=None):  self.addToSection('infer', inferLines, params)
    def addToTrain(self, trainLines, params=None):  self.addToSection('train', trainLines, params)
    
    # ******************************************************************************************************************
    def methodDefined(self, methodName):
        return (methodName in self.definedMethods)
        
    # ******************************************************************************************************************
    def defineMethod(self, methodName):
        self.definedMethods.add(methodName)
        
    # ******************************************************************************************************************
    def getL2LossFactor(self, layer):
        l2LossFactor = "0"
        if self.fbModel.regFactor > 0:
            for pa in layer.postActivations:
                if pa.name=='L2R':
                    if pa.factor>0:
                        factor = pa.factor if pa.factor != 1.0 else self.fbModel.regFactor
                        l2LossFactor = str(factor) + " if isTraining else 0"
        return l2LossFactor
    
    # ******************************************************************************************************************
    def export(self, tfFolderPath, **kwargs):
        # Extract main arguments:
        quiet = kwargs.get("quiet", False)
        modelFileName = kwargs.get('modelFileName', 'TfModel.py')
        paramsFileName = kwargs.get('paramsFileName', 'Params.npz')
        self.runQuantized = kwargs.get('runQuantized', False)

        t0 = time.time()
        if not quiet:
            myPrint('\nExporting to TensorFlow model "%s" ... '%(tfFolderPath))

        if not os.path.exists(tfFolderPath):  os.makedirs(tfFolderPath)
        
        tfFileName = tfFolderPath + '/' + modelFileName
        self.tfFile = open(tfFileName, "w")
        self.writeHeader(modelFileName)
        
        self.initStr = ""
        self.methodsStr = ""
        self.graphStr = "    def buildGraph(self, isTraining=False):\n"
        self.inferStr = "    def infer(self, samples):\n"
        self.trainStr = "    def trainBatch(self, optimizeOp, batchSamples, batchLabels):\n"

        self.addMethod(("def makeVariable(self, name, shape, stddev, codebookSize=0, trainable=True):",
                        "    if codebookSize == 0:",
                        "        if stddev == 0:",
                        "            var = tf1.get_variable(name, shape, tf.float32, initializer=tf.zeros_initializer(), trainable=trainable)",
                        "        elif stddev == 1:",
                        "            var = tf1.get_variable(name, shape, tf.float32, initializer=tf.ones_initializer(), trainable=trainable)",
                        "        else:",
                        "            initVal = tfr.truncated_normal(shape, mean=0, stddev=stddev, seed=SEED)",
                        "            var = tf1.get_variable(name, initializer=initVal, trainable=trainable)",
                        "    else:",
                        "        codebook = tf1.get_variable(name+'Codebook', [codebookSize], tf.float32,",
                        "                                    initializer=tf.zeros_initializer(), trainable=trainable)",
                        "        indexes = tf1.get_variable(name+'Indexes', shape, tf.int32,",
                        "                                   initializer=tf.zeros_initializer(), trainable=False)",
                        "        var = tf1.gather(codebook, indexes, name=name)",
                        "    return var",
                        ""))

        self.addMethod(("def loadParameters(self, paramsFilePath):",
                        "    rootDic = np.load(paramsFilePath)",
                        "",
                        "    variables = tf.compat.v1.global_variables()",
                        "    ops = []",
                        "    feedDic = {}",
                        "    for var in variables:",
                        "        varName = var.name[:-2]",
                        "        if varName in ['GlobalStep']: continue",
                        "        if varName not in rootDic:") + ((
                        "            print('Variable \"%s\" was not initialized!'%(varName))",
                        "            continue") if self.runQuantized else (
                        "            if ((varName+'Codebook') in rootDic) and ((varName+'Indexes') in rootDic):",
                        "                paramVal = rootDic[varName+'Codebook'][ rootDic[varName+'Indexes'] ]",
                        "            else:",
                        "                print('Variable \"%s\" was not initialized!'%(varName))",
                        "                continue")) + (
                        "        else:",
                        "            paramVal = rootDic[ varName ]",
                        "        ops += [ var.initializer ]",
                        "        feedDic[ var.initializer.inputs[1] ] = paramVal",
                        "    self.session.run( ops, feedDic )",
                        ""))

        for l,layer in enumerate(self.fbModel.layers):
            if not quiet:
                myPrint('    Now processing "%s" (layer %d of %d) ...           \r'%(layer.scope, l+1,
                                                                                     self.fbModel.layers.count), False)
            layer.buildTf(self)
            
        self.addToInit(("",
                        "self.buildGraph()",
                        "if train:",
                        "    self.buildGraph(True)",
                        "    self.globalStep = tf.Variable(0, trainable=False, name='GlobalStep')",
                        "elif paramsFilePath is None:",
                        "    paramsFilePath=exportPath+'/%s'"%(paramsFileName),
                        "",
                        "initializer = tf.compat.v1.global_variables_initializer()",
                        "self.session = tf1.Session()",
                        "self.session.run( initializer )",
                        "if paramsFilePath is not None: self.loadParameters(paramsFilePath)",
                        ""))

        self.tfFile.write(self.initStr)
        self.tfFile.write(self.methodsStr)
        self.tfFile.write(self.graphStr)
        self.tfFile.write(self.inferStr)
        self.tfFile.write(self.trainStr)

        # If present, it must be a list of strings one for each class. We add this to the end of the
        # file.
        classNames = kwargs.get('classNames', None)
        if classNames is not None:
            classNamesLines = ["def getClassNames(self):"]
            tfStr = "    return ["
            while len(classNames)>0:
                tfStr += ','.join(["'%s'"%(s.replace("'", "\\'")) for s in classNames[:8]])
                classNames = classNames[8:]
                if len(classNames)>0:   tfStr += ','
                else:                   tfStr += ']'
                classNamesLines += [tfStr]
                tfStr = '            '
            classNamesLines += ['']
            self.tfFile.write( getStr('    ', tuple(classNamesLines)) )

        self.tfFile.close()
        
        if not quiet:
            myPrint('    Processed all %d layers.                                     '%(self.fbModel.layers.count))
            
        # Now write the network parameters to a numpy compressed file.
        if not quiet:   myPrint('    Creating parameters file "Params.npz" ... ', False)
        nameNetParams = self.fbModel.layers.getNameNetParams()
        rootDic = {}
        nameList = []
        for name,netParam in nameNetParams:
            if netParam.codebook is None:
                nameList += [name]
                rootDic[ name ] = netParam.value()
            else:
                nameList += [name+'Codebook']
                nameList += [name+'Indexes']
                rootDic[ name+'Codebook' ] = netParam.codebook
                rootDic[ name+'Indexes' ] = netParam.rawVal

        rootDic['netParams'] = nameList    # This gives the order of parameters
        np.savez_compressed(tfFolderPath + '/' + paramsFileName, **rootDic)
        if not quiet:   myPrint('Done.\nDone.')
