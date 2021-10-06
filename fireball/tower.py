# Copyright (c) 2019-2020 InterDigital AI Research Lab
"""
This file contains the implementation of "Tower" class used by Fireball
models for utilizing multiple GPU's while training.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 05/10/2019    Shahab Hamidi-Rad       Created the file.
# 08/10/2020    Shahab                  Moved some of the input/output functionality to the new input/output layers.
# **********************************************************************************************************************
import numpy as np

import tensorflow as tf
try:    import tensorflow.compat.v1 as tf1
except: tf1 = tf

from .layers import checkNumeric
from .netparam import NetParam

# **********************************************************************************************************************
# Tower class
# **********************************************************************************************************************
class Tower:
    """
    Tower
    This class encapsulates the data and functionality for a tower. A tower is a representation of a processing
    unit that can be a cpu or a gpu card.
    """

    # ******************************************************************************************************************
    def __init__(self, owner, towerNo=0, deviceName='/cpu:0', initValues=None):
        self.owner = owner
        self.towerNo = towerNo
        self.deviceName = deviceName
        self.initValues = initValues
        
        self.tfNetParams = []
        self.allTrainable, self.nonTransferred = [], []
        self.tfLayerOutputs = {'training':[], 'inference':[]}

        self.tfInferPrediction = None
        self.tfBatchPrediction = None
        self.evalResults = None

    # ******************************************************************************************************************
    def makeTfVariables(self):
        self.tfNetParams = []
        initIndex = 0
        for layer in self.owner.layers:
            layerInitValues = None
            if self.initValues is not None:
                if initIndex < len(self.initValues):
                    layerInitValues = self.initValues[initIndex:]
            layer.paramIndex = initIndex
            layerVars = layer.makeVars(layerInitValues)
            initIndex += len(layerVars)
            self.tfNetParams += layerVars

        for netParam in self.tfNetParams:
            if not netParam.trainable:  continue
            self.allTrainable += [ netParam.tfVariable ]
            if netParam.initialized:    continue
            self.nonTransferred += [ netParam.tfVariable ]
    
    # ******************************************************************************************************************
    def freezeLayerParams(self, layers):
        firstFreezedLayer = lastFreezedLayer = None
        if type(layers) == tuple:
            assert len(layers)==2, "The layers tuple must have exactly 2 item, but it has %d."%(len(layers))
            firstFreezedLayer, lastFreezedLayer = layers
        
        numFreezed = 0
        for layer in self.owner.layers:
            if type(layers) == list:
                if layer.scope not in layers:           continue
            elif firstFreezedLayer is not None:                     # Not started yet
                if layer.scope != firstFreezedLayer:    continue    # Still not started
                firstFreezedLayer = None                            # Just started
            elif lastFreezedLayer is None:              continue    # already ended
            elif layer.scope == lastFreezedLayer:                   # Just ended
                lastFreezedLayer = None
            
            numFreezed += len(layer.netParams)
            for param in layer.netParams:   param.trainable = False # Freeze all layer params

        assert firstFreezedLayer is None, "Could not find the specified first layer \"%s\"!"%(firstFreezedLayer)
        assert lastFreezedLayer is None, "Could not find the specified last layer \"%s\"!"%(lastFreezedLayer)
        assert numFreezed>0, "Could not find any layer to freeze!"
        
        newTrainables = []
        for netParam in self.tfNetParams:
            if netParam.trainable: newTrainables += [ netParam.tfVariable ]
        self.allTrainable = newTrainables
        return numFreezed
        
    # ******************************************************************************************************************
    def makePlaceHolders(self):
        self.samplesPlaceholders = self.owner.layers.input.makePlaceholders()
        self.labelsPlaceholders = self.owner.layers.output.makePlaceholders()

    # ******************************************************************************************************************
    def getNetOutput(self, training):
        lastLayerOut = self.samplesPlaceholders
        self.batchNormMoments = []
        
        for l,layer in enumerate(self.owner.layers):
            lastLayerOut = checkNumeric(lastLayerOut, "Input to \"%s\" contains NAN/INF!!"%(layer.scope))

            layerInput = layer.getInput(lastLayerOut, training)
            lastLayerOut, additionalInfo = layer.buildGraph( layerInput, training, self.labelsPlaceholders)
            
            if additionalInfo is None: continue
            
            if layer.supportsEval:
                # Output is regression to multi-dimensional tensors. In this case we
                # also calculated evaluation results which are returned in "additionalInfo"
                assert not training
                self.evalResults = additionalInfo
            else:
                # In this case the additionalInfo contains the batch normalization moments
                self.batchNormMoments += additionalInfo

        return lastLayerOut

    # ******************************************************************************************************************
    def makeTrainGraph(self):
        self.tfLoss = self.getNetOutput(training=True)
        
        # L2 Regularization on specified layers
        tfL2Reg = None
        if self.owner.regFactor > 0:
            with tf.name_scope('L2Reg'):
                for layer in self.owner.layers:
                    if layer.l2Loss is None: continue
                    if tfL2Reg is None:     tfL2Reg = layer.l2Loss
                    else:                   tfL2Reg += layer.l2Loss
                if tfL2Reg is not None:
                    tfL2Reg = self.owner.regFactor * tfL2Reg

        if tfL2Reg is not None: self.tfLoss += tfL2Reg

        with tf.name_scope('AbsGrads'):
            grads = tf.gradients(self.tfLoss, self.allTrainable)
            # Use zero grad for non-trainable variables for which the gradients function above returns None
            self.tfAbsGrads = [ tf.zeros_like(self.allTrainable[i]) if g is None else tf.abs(g) for i,g in enumerate(grads) ]
#            self.tfAbsGrads = [ tf.zeros_like(self.tfNetParams[i]) if g is None else tf.square(g) for i,g in enumerate(grads) ]

    # ******************************************************************************************************************
    def makeInferGraph(self):
        self.tfInferPrediction = self.getNetOutput(training=False)

    # ******************************************************************************************************************
    def getGrads(self, optimizer):
        with tf.device( self.deviceName ):
            with tf.name_scope( 'Tower%d'%(self.towerNo) ):
                with tf.name_scope('Inputs'):
                    self.makePlaceHolders()

                with tf1.variable_scope('Variables',reuse=tf1.AUTO_REUSE):
                    self.makeTfVariables()

                with tf.name_scope('Inference'):
                    self.makeInferGraph()

                with tf.name_scope('Training'):
                    self.makeTrainGraph()
                    with tf.name_scope('OptGrads'):
                        myGrads = optimizer.compute_gradients(self.tfLoss, self.allTrainable)
                        myGradsNT = None if len(self.nonTransferred)==0 else \
                                            optimizer.compute_gradients(self.tfLoss, self.nonTransferred)

#        myGrads = [ (tf.zeros_like(self.allTrainable[i]), g[1]) if g[0] is None else g for i,g in enumerate(myGrads) ]
#        myGradsNT = [ (tf.zeros_like(self.nonTransferred[i]), g[1]) if g[0] is None else g for i,g in enumerate(myGradsNT) ]
        return (myGrads, myGradsNT)

    # ******************************************************************************************************************
    def __repr__(self):
        retStr = '\n%s Instance:'%(self.__class__.__name__)
        retStr += '\n    Tower Num.:        ' + str(self.towerNo)
        retStr += '\n    Device Name:       ' + self.deviceName
        return retStr
