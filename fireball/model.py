# Copyright (c) 2017-2020 InterDigital AI Research Lab
"""
The core implementation of the "fireball" library. This file implements the
"Model" class. A model can be loaded from a file or created from scratch.
The network structure can be specified using a short-form language that
defines different layers of the network in a text string.

Please refer to the "Layers" class for more details about the types of
layers currently supported by Fireball.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 07/28/2017    Shahab Hamidi-Rad       Created the file.
# 07/10/2018    Shahab                  Major modifications:
#                                         * Support for 1D Convolution
#                                         * Tensorboard support
#                                         * Layer specification by simple text strings.
# 03/04/2019    Shahab                  Added support for LR/LDR layers (reducing number of parameters).
# 05/03/2019    Shahab                  Added support for Quantization and Arithmetic Coding.
# 05/10/2019    Shahab                  Added support for TensorFlow Towers and parallel training.
#                                           Integrated the code for Multi-threaded batch providers (See ImageNet).
# 05/20/2019    Shahab                  Reviewed and updated the documentation of all functions.
# 05/21/2019    Shahab                  Added support for specifying, accessing, and modifying the output layer.
# 05/24/2019    Shahab                  Added support for low rank convolutional layers(LRC).
# 06/11/2019    Shahab                  Added support for:
#                                         * Specific Padding values besides the existing 'same' and 'valid' cases.
#                                         * Non-square kernels, strides, and padding  rank convolutional layers(LRC).
#                                         * Average Pooling.
#                                         * New Language with more flexibility.
#                                         * Concept of Post-Activation operators that are applied to the
#                                           output of the activation function. Examples include max/average
#                                           pooling. In future we can have more of these.
#                                         * Added support to get the output of each sub-layer inside a layer.
# 06/18/2019    Shahab                  Added support for:
#                                         * UpSampling Post Activation operator.
#                                         * Any-Shape output for regression problems.
# 07/22/2019    Shahab                  Added Basic Support for ONNX export and import.
#                                         * Export every type of layer except Conv1D and LDR.
#                                         * Import FC, Conv, LR, LRC laters.
# 07/25/2019    Shahab                  Changed the name of this library to "fireball" and this file to model.
# 09/13/2019    Shahab                  Some modifications to support python 3.
# 03/09/2020    Shahab                  Changed interaction with datasets. Now receiving dataset objects instead
#                                           of the actual data. See also the "datasets" folder.
# 03/16/2020    Shahab                  Added support for quantized models. You can now use the class method
#                                           "quantizeModel" function to quantize a model and save it in an .fbm
#                                           file. The file can be loaded and retrained. (Backprop through codebooks)
# 03/22/2020    Shahab                  Completed the SSD object detection implementation.
# 08/12/2020    Shahab                  Fireball 1.3 features:
#                                         * Support for new layers: BERT, LN, Input, and output layers.
#                                         * LayerInfo now must always have an input and an output layer. A lot
#                                           of functionality was moved to the input/output layers. Most of
#                                           customization for different tasks are performed in these layers.
#                                         * Support for Transfer learning. You can import params values for only
#                                           some of parameters and initialize the rest (Non-Transferred params)
#                                           randomly.
#                                         * You can start the training with only non-transferred parameters and
#                                           then switch to all trainable parameters after a specified number of
#                                           batches. (See "trainAllBatch")
#                                         * Changed the format of fbm files. Instead of using npz files, we are
#                                           now using Fireball's own format defined in "fnjfile.py".
#                                         * Added support for tokenization in the "textio.py" file.
#                                         * Added support for SQuAD dataset in the "datasets/squad.py" file.
#                                         * Changed piecewise learning rate from epoch-based to batch-based.
#                                         * Moved all evaluation functionality to the corresponding dataset objects.
#                                         * Added support for GLUE datasets in the "datasets/glue.py" file.
#                                         * Updated CoreML export functionality with BERT and input/output layers.
# 11/29/2020    Shahab                  Fireball 1.4 features:
#                                         * Tensorflow export for all Fireball layers and models including
#                                           Low-rank models and codebook-quantized models. See the fb2tf.py file
#                                           for more information.
#                                         * ONNX export for all Fireball layers and models including Low-rank
#                                           models and codebook-quantized models. See the fb2onnx.py file for
#                                           more information.
#                                         * Re-structured the implementation of CoreML export for all Fireball
#                                           layers and models including Low-rank models and codebook-quantized models.
#                                           See the fb2coreml.py file for more information.
#                                         * Support for user-defined loss functions. The Model constructor now
#                                           receives a new argument 'lossFunction'. See the documentation for this
#                                           argument below for more information. Fireball even saves the user-defined
#                                           loss function in the fbm file. So, the receiver of the file does not
#                                           need to know the user-defined loss function. It just becomes part of
#                                           the model.
#                                         * Support for user-defined layers. Use the "registerLayerClass" function
#                                           to register your own layers and integrate them into Fireball.
#                                         * Support for Netmarks. See the layers.py for more information.
# 01/28/2021    Shahab                  Fireball 1.5 features:
#                                         * Support for pruning the models. Pruning can be applied on a model or on a
#                                           low-rank model. Re-training (AKA fine-tuning) is also supported for the
#                                           pruned models.
#                                         * Support for quantizing pruned models. Also imported some of the
#                                           improvements made for MPEG NNR from NCTM. Trained quantization is now
#                                           supported for pruned models.
#                                         * Support for Warm-up learning rate scheduling. See the explanation for the
#                                           "learningRate" argument of the Model's __init__ method for more info.
# 10/11/2021    Shahab                  Fireball 1.5.1 features:
#                                         * First public version.
#                                         * Added support for downloading from model zoo.
# 05/12/2022    Shahab                  Fireball 2.0.0 features:
#                                         * createLrModel now receives additional options with "kwargs".
#                                         * Added the new method createPrunedModel
# **********************************************************************************************************************
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import time
import json
import urllib.request as urlreq
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fireball
import numpy as np

# There are problems with profobuf versions between ONNX and TensorFlow. ONNX must be
# imported before TensorFlow! Remove the following 2 lines when this problem hopefully
# gets resolved in a future version of ONNX.
try:    import onnx
except: pass

import tensorflow as tf
try:    import tensorflow.compat.v1 as tf1
except: tf1 = tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from . import ldr           # Implementation of LR/LDR functionality used to reduce the number of parameters
from .tower import Tower    # Encapsulating the graph information and behavior for each GPU device
from .utils import *        # Utility functions
from .layers import Layer,Layers,Block
from .netparam import NetParam
from .printutils import myPrint
from .fnjfile import loadFNJ, saveFNJ

# **********************************************************************************************************************
LEARNING_DECAY_RATE = .95       # Every Model::lrDecayStep batch, multiply current learning rate by
                                # LEARNING_DECAY_RATE

# **********************************************************************************************************************
class Model:
    r"""
    The implementation of Fireball Model.
    
    A General Purpose Neural Network for Classification and Regression problems. Fireball is built on top of Tensorflow 1.x.
    """
    quiet = False   # Set to True to stop printing in all functions of this class.
    class config:
        r"""
        The configuration information used for object detection networks such as SSD.
        """
        maxDetectionsPerImage=200   # For inference, use 20
        maxDetectionPerClass=100    # For inference, use 20
        iouThreshold=.45
        scoreThreshold=.01          # For inference, use 0.50

    # ******************************************************************************************************************
    def __init__(self,
                 name='FireballModel', layersInfo=None,
                 trainDs=None, testDs=None, validationDs=None,
                 batchSize=None, numEpochs=10, regFactor=0.0, dropOutKeep=1.0,
                 learningRate=.01, optimizer=None, lossFunction=None, blocks=[],
                 modelFilePath=None, saveModelFileName=None, savePeriod=None, saveBest=False,
                 gpus=None, netParams=None, trainingState=None):
        r"""
        Initialize all the parameters and then build a TensorFlow graph based on the parameters.
        
        Parameters
        ----------
        name : str
            An optional name for the network.
                
        layersInfo : str
            If contains information for each hidden layer. The string should be in the format explained in the "Layers" class.
                            
        trainDs : dataset object, Derived from "BaseDSet" class
            Must conform to the dataset object. Used only for training. For more information about the dataset classes, please refer to the "datasets" directory.
            
        testDs : dataset object, Derived from "BaseDSet" class
            Must conform to the dataset object. Used for testing. For more information about the dataset classes, please refer to the "datasets" directory.
            
        validationDs : dataset object, Derived from "BaseDSet" class
            Must conform to the dataset object. Used for evaluation and hyper-parameter search.
            
        batchSize : int
            The batch size used in each iteration of training. If not specified, the batch size of "trainDs" is used for training.
            
        numEpochs : int
            Total number of Epochs to train the model.
            
        regFactor : float
            Regularization Factor. If this is zero, then L2 regularization is disabled for all layers. Otherwise, if this is non-zero, then L2 regularization is enabled. In this case:
            
                * If a factor is specified in the L2R post-activation, the specified value is used for that layer.
                * Otherwise, if the factor is not specified, then this value is use for the L2 regularization factor for that layer.
            
        dropOutKeep : float, default = 1.0
            The probability of keeping results for dropout. dropRate is (1.0 - dropOutKeep). This value can be one of the following:
            
                * If this value is 1 (default), then all DO layers use their specified rate. If a rate was not specified for a DO layer, Dropout is disabled for that later.
                * If this values is non-zero and less than 1, then all DO layers use their specified rate. If a rate was not specified for a DO layer, its rate is set to "1.0-dropOutKeep".
                * If this value is 0.0, then Dropout is disabled globally in the whole network.

        learningRate : tuple, list, float, or None
            - None: This is used when the model is not used for training. (trainDs should also be None in this case)
            - tuple: If it is a tuple (start, end, [warmUp]), the learning rate starts at "start" value and exponentially decays to "end" value at the end of training. The optional "warmUp" value gives the number of "Warm-up" batches at the beginning of training. During warm-up phase, the learning rate increases linearly from 0 to the "start" value.
            - float: If it is a single floating point value, the learning rate is fixed during the training.
            - list of tuples: If it is a list of the form: [(0,lr1),(n2,lr2), ...,(nN,lrN)], learning rate is changed based on a piecewise equation. It starts at "lr1" and stays constant until it reaches batch "n2". Then the learning rate becomes "lr2" until it reaches batch "n3" where it changes to "lr3", and so on. The values "n2", "n3", ..., "nN" must monotonically increase. The last learning rate is used until the end of training. Note that "n1" must always be equal to 0 specifying initial learning rate.

                * **Example**::
                
                    learningRate=[(0,.01),(300,.005),(400, .001)]
                    
                  Learning rates for batches:
                        
                        ==========================   =============
                        Batches                      Learning Rate
                        ==========================   =============
                        0 to 299                     0.01
                        300 to 399                   0.005
                        400 to the end of training   0.001
                        ==========================   =============

                One special case is when a tuple of the form (b, 'trainAll') is
                included. This is used when doing transfer learning. In this
                case we train only the non-transferred parameter before batch
                'b' and all trainable parameters after batch 'b'. This does not
                change the learning rate behavior.
                
                * **Example**::

                    learningRate=[(0,.1),(200,'trainAll'),(300,.05),(400, .01)]
                    
                  Schedule of batches:
                    
                        ==========  ==============  ====================
                        Batches     Learning Rate   Trained Parameters
                        ==========  ==============  ====================
                        0 to 199    0.1             Only non-transferred
                        200 to 299  0.1             All trainable
                        300 to 399  0.05            All trainable
                        400 to end  0.01            All trainable
                        ==========  ==============  ====================
                        
                If the first tuple is of the form (0, "WarmUp"), then it
                means the training starts with "w" batches of Warm-up
                before continuing with the piecewise. The number of Warm-up
                batches "w" is specified by the next tuple in the list.
                
                * **Example**::

                    learningRate=[(0,'WarmUp'), (100,.1), (200,'trainAll'), (300,.05), (400, .01)]
                    
                  Schedule of batches: (w=100 in this case)
                    
                        ==========  ==============  ====================
                        Batches     Learning Rate   Trained Parameters
                        ==========  ==============  ====================
                        0 to 99     b*(0.1)/100     Only non-transferred
                        100 to 199  0.1             Only non-transferred
                        200 to 299  0.1             All trainable
                        300 to 399  0.05            All trainable
                        400 to end  0.01            All trainable
                        ==========  ==============  ====================

        optimizer : str
            The type of optimizer used for training. Available options are: 'GradientDescent', 'Adam' (Default), and 'Momentum'. If not specified, "Momentum" is used for classification and "Adam" for regression problems.
            
        lossFunction : function
            A function that is used to calculate the loss in the training mode. This is used to define a customized loss function. This is used by the model to calculate the loss during the training. The function takes the following arguments:
            
                * layers: A "Layers" object as defined in the "Layers.py" file. This keeps a list of all the layers in the model that may be used for the calculation of the loss.
                * predictions: The output(s) of the network just before the output layer. This is a tuple containing all outputs of the network.
                * groundTruths: The batch of labels used for the training step. This is a tuple containing all label objects. The tuple usually contains the placeholders created in the Layer's "makePlaceholders" function.
            
        blocks : list
            A list of Block objects or *blockInfo* strings that extend fireball's predefined layers and can be used in the *layersInfo* string. For more information about how blocks work, please refer to :ref:`BLOCKS`.
            
        modelFilePath : str
            The path to the model file used to load this model. Only set when the makeFromFile is used to create a model.
            
        saveModelFileName : str
            The file name to use when saving the trained network information. A training session can be resumed using information in the saved file.
            
            If the "saveBest" argument is True, then this file name is also used to create another file that saves the network with best results so far during the training. If this argument is None, then the "savePeriod" and "saveBest" arguments are ignored and the network information is not saved during the training. If this argument is True, then the file name "SavedModel" will be used to save the training information.
            
        savePeriod : int
            The number of epochs before saving the current training state. The training can be resumed using the files saved during the training.
            
            Ignored if "saveModelFileName" is None. If this is set to None or 0, then the trained network is saved only once at the end of training. In this case if the training is interrupted for any reason, the training restarts from the begining in the next session (all training info is lost).
            
        saveBest : Boolean
            If true, the network with best results so far during the training will be saved in a separate file. This is useful when the network performance degrades as more training epochs are processed. Ignored if "saveModelFileName" is None.
            
        gpus : Number, List, or str
            If this is a list, it should contain the integers specifying the GPU devices to be used. For example [0,3] means this class will use the "/gpu:0" and "/gpu:3" devices during the training and inference.
            
            If this is a string, it can be "All" or "Half" which means use all or half of the detected gpu devices correspondingly. Or it can be a comma delimited list of gpu numbers for example: "1,3".
            
            If this is an integer, that number would specify the only GPU device that will be used.
            
            If this is None, this class uses half of the detected GPUs. (Same as passing the "Half" string)
            
            Use [-1] to disable GPU usage and run on CPU.
            
        netParams : list
            This parameter is only used internally when this constructor function is called from "makeFromFile". It is a list of NetParam objects.
            
        trainingState : dict
            This parameter is only used internally when this constructor function is called from "makeFromFile". It contains the training state that can be used to resume an interrupted training session.
        """
        self.trainDs = trainDs
        self.testDs = testDs
        self.validationDs = validationDs
        
        self.layers = Layers(layersInfo, [Block[b] if type(b)==str else b for b in blocks], self)
        self.layers.setAllShapes()

        self.numClasses = 0
        if self.layers.output.name == 'CLASS':
            self.numClasses = self.layers.output.numClasses
        assert self.numClasses!=1, "The number of classes cannot be 1!"
        
        self.optimizer = optimizer
        if optimizer is None:   self.optimizer = "Momentum" if self.numClasses>0 else "Adam"
        self.lossFunction = lossFunction
            
        self.name = name
        self.trainingState = trainingState
        
        # Setting GPUs to use:
        if type(gpus) == list:              self.gpus = gpus
        elif type(gpus) == tuple:           self.gpus = list(gpus)
        elif type(gpus) == str:
            if gpus.lower() == 'all':       self.gpus = getCurrentGpus()
            elif gpus.lower() == 'half':    self.gpus = getCurrentGpus(True)
            else:
                self.gpus = [ int(x) for x in gpus.split(',') ]
        elif type(gpus) == int:             self.gpus = [gpus]
        else:                               self.gpus = getCurrentGpus(False)[-1:] # Pick last GPU or CPU if none

        self.afterBatch = None
        self.afterEpoch = None

        self.batchSize = batchSize
        if (batchSize is None) and (trainDs is not None):   self.batchSize = trainDs.batchSize
        self.numEpochs = numEpochs
        self.regFactor = regFactor
        self.dropRate = 1.0 - dropOutKeep   # The global dropout rate used by "DO" layers with unspecified rate.
        
        self.learningRate = learningRate    # This is used to save/load from files
        self.learningRateInit = 0.01
        self.lrDecayStep = 0
        self.learningRatePieces = None
        self.learningRateWarmUp = 0
        self.trainAllBatch = 0              # Train only non-transferred params before this batch, all params after.
        if learningRate is None:
            assert self.trainDs is None, "learningRate must be a specified when a training dataset is provided!"
        elif type(learningRate) == tuple:
            # Exponential decay:
            assert len(learningRate) in [2,3], "Need two or three entries for Exponential Decay Learning Rate!"
            if len(learningRate)==2:    learningRate += (0,)
            self.learningRateInit, self.learningRateMin, self.learningRateWarmUp = learningRate
            assert self.learningRateInit>self.learningRateMin, "Initial value should be larger than final value!"
            self.learningRatePieces = None
            
            # Calculating the decay step size:
            if self.trainDs is not None:
                self.lrDecayStep = (((self.trainDs.numSamples//self.batchSize)*self.numEpochs)//
                                    (np.log(self.learningRateMin/self.learningRateInit)/np.log(LEARNING_DECAY_RATE)))
                if self.lrDecayStep == 0: self.lrDecayStep = 1
            
        elif type(learningRate) == list:
            # piecewise learningRate:
            assert len(learningRate)>1, "Need at least two entries for Piecewise Learning Rate!"
            assert type(learningRate[0]) == tuple, "Piecewise Learning Rate must be a list of tuples!"
            assert len(learningRate[0]) == 2, "Piecewise Learning Rate must be a list of 2-tuples!"
            assert learningRate[0][0] == 0, "First entry in the list must specify the initial learning rate (0, <initRate>)!"
            
            self.learningRateWarmUp = 0
            if type(learningRate[0][1])==str:
                assert learningRate[0][1].lower() == "warmup", "Invalid syntax! Did you mean (0, 'WarmUp') for the first tuple?"
                self.learningRateWarmUp = learningRate[1][0]
            
            for i in range(1, len(learningRate)):
                assert learningRate[i][0] >= learningRate[i-1][0], "Batch numbers must be in increasing order! " \
                                                                   "(%d -> %d)"%(learningRate[i-1][0],learningRate[i][0])

            # Note: Boundaries includes the initial 0.
            lrBoundaries = [ batch for batch,val in learningRate if type(val)!=str ]
            lrValues =     [ val   for batch,val in learningRate if type(val)!=str ]
            
            for batch,val in learningRate:
                if type(val)==str and val.lower()=='trainall':    self.trainAllBatch = batch
                
            self.learningRateInit = lrValues[0]
            self.learningRatePieces = (lrBoundaries, lrValues)
            self.learningRateMin = None
        elif type(learningRate) in [float, np.float32, np.float64]:
            # Fixed Learning Rate:
            self.learningRateInit = self.learningRateMin = float(learningRate)
            self.learningRatePieces = None
        else:
            raise ValueError("learningRate must be a tuple, a list of tuples, or a floating point number!")

        self.modelFilePath = modelFilePath
        self.saveModelFileName = saveModelFileName
        self.paramNames = None
        if netParams is None:   # If netParams is given, we have already handled this in "makeFromFile" function
            if type(saveModelFileName) is bool:        # Save the model in <curPath>/SavedModel
                if saveModelFileName==True:
                    self.saveModelFileName = os.path.dirname(sys.modules['__main__'].__file__) + 'SavedModel.fbm'
            if saveModelFileName is not None:
                if self.saveModelFileName[-4:] != '.fbm':   self.saveModelFileName += '.fbm'
                _, netParams, _, self.trainingState  = Model.loadModelFrom(self.saveModelFileName)

        self.saveBest = saveBest
        self.savePeriod = self.numEpochs if ((savePeriod is None) or (savePeriod==0)) else savePeriod
        
        with tf.name_scope(self.name):
            self.buildTowers(netParams)
        self.numParams = self.layers.getNumParams()
        
        if (trainDs is not None) and (self.layers.output.name == 'OBJECT'):
            trainDs.setAcnchorBoxes( self.layers.output.anchorBoxes )

        self.minLoss = None
        self.maxLoss = None
        self.minLossCount = None
        self.session = None
        self.trainTime = 0.0
        self.sumAbsGrads = None
        self.quiet = False
        self.bestMetric = None
        self.bestEpochInfo = None

    # ******************************************************************************************************************
    @classmethod
    def downloadFromZoo(cls, modelName, destFolder, modelType=None):
        r"""
        A class method used to download a model from an online Fireball model zoo.
        
        Parameters
        ----------
        modelName: str
            A string containing the name of the model. If the "modelType" parameter is not
            provided, then this name must include the file extension to help identify the type
            of the model.
            
        destFolder: str
            The folder where the downloaded model file is saved.
            
        modelType: str
            The type of the model. Currently the following types are supported:
            
                * 'Fireball': Fireball models. (Extension: fbm)
                * 'CoreML': Models exported to CoreML ready to be deployed to iOS. (Extension: mlmodel)
                * 'ONNX': Models exported to ONNX. (Extension: onnx)
                * 'NPZ': Numpy 'npz' files containing the model information. (Extension: npz)
        """
        type2ext = {'Fireball': "fbm", 'CoreML': 'mlmodel', 'NPZ':'npz', 'ONNX':'onnx'}
        ext2Type = {v:k for k,v in type2ext.items()}

        if modelType is None:
            # Try getting model type from extension:
            ext = os.path.splitext(modelName)[-1][1:].lower()
            if ext not in ext2Type:
                raise Exception("Unknown model type '%s'"%(ext))
            modelType = ext2Type[ext]
        else:
            ext = type2ext[modelType]

        if destFolder[-1] != '/': destFolder += '/'
        if not os.path.exists(destFolder):
            print('Creating folder "%s" ...'%(destFolder))
            os.makedirs(destFolder)
        elif os.path.isfile(destFolder):
            raise ValueError("The destination must be a directory!")

        modelFileName = modelName if modelName[-len(ext)-1:].lower() == ('.'+ext) else (modelName + "." + ext)
        destFilePath = destFolder + modelFileName
        
        if os.path.exists(destFilePath):
            return
        
        locInfoUrl = "https://interdigitalinc.github.io/Fireball/LocInfo.yml"
        locInfo = yaml.safe_load(urlreq.urlopen(locInfoUrl).read())
        for location in locInfo['modelLocations']:
            if location[-1] != '/': location+='/'
            modelUrl = "%s%s/%s"%(location, modelType, modelFileName)
            try:
                print('Downloading from "%s" ...'%(modelUrl))
                urlreq.urlretrieve(modelUrl, destFilePath)
                print('  Success!')
                break
            except:
                print('  Failed!')
                continue

    # ******************************************************************************************************************
    @classmethod
    def loadModelFrom(cls, fileName):
        r"""
        This is a class method that is used to read the network information from a file and return the results.
        
        Parameters
        ----------
        fileName : str
            The name of the file containing the network information. This can be:
            
                * A "Model" file with 'fbm' extension with all network information in a single numpy npz file,
                * A file with 'fbmc' extension containing the quantized and compressed network information,

        Returns
        -------
        graphInfo : dict
            A dictionary containing information about structure, inout, output, etc. of the network.
        netParams : list
            The list of NetParam objects for all network parameters (i.e. weights and biases)
        trainInfo : dict
            A dictionary containing training information.
        trainState : dict
            A dictionary containing the last training state such as the epoch number, learning rate, etc.
        """
        def bytes2Str(bytesStr):
            if type(bytesStr) in [bytes, np.bytes_]: return bytesStr.decode("utf-8")
            return bytesStr

        if os.path.exists(fileName)==False:     return None, None, None, None
        if os.stat(fileName).st_size == 0:      return None, None, None, None
            
        graphInfo, netParams, quantInfo, trainInfo, trainState = 5*[None]
        graphInfo = {
                            'name': fileName,
                            'producer': "",
                            'version': fireball.__version__,
                            'doc': 'Imported From "%s".'%(fileName),
                            'layersInfo': None,
                            'inShape': None,
                            'outShape': None,
                            'numClasses': 0
                    }

        trainInfo = {
                            'batchSize': None,
                            'numEpochs': 10,
                            'regFactor': 0.0,
                            'dropOutKeep': 1.0,
                            'learningRate': None,
                            'optimizer': None,
                            'savePeriod': None,
                            'saveBest': False
                    }

        if fileName[-5:].lower() == '.onnx':
            # ONNX Format
            import fbonnx
            return fbonnx.importFromOnnx(fileName, graphInfo, trainInfo)
        
        if (fileName[-3:].lower() == '.h5') or (fileName[-5:].lower() == '.hdf5'):
            # Keras model Format
            # The ordered names of parameters must be in a file with the same name and "yaml" extension.
            import h5py
            import yaml
            namesFileName = fileName[:-3] + '.yaml'
            if fileName[-5:].lower() == '.hdf5':
                namesFileName = fileName[:-5] + '.yaml'

            h5File = h5py.File(fileName, mode='r')
            modelParams = h5File.get('model_weights', h5File)

            netParams = []
            with open(namesFileName, 'r') as namesFile:
                orderedNames = yaml.load(namesFile, Loader=yaml.FullLoader)
            for layerName, layerInfo in orderedNames:
                for weightName in layerInfo:
                    weightInfo = None
                    if type(weightName)==list: weightName, weightInfo = weightName
                    if type(weightInfo)==list:
                        for subWeightName in weightInfo:
                            if type(subWeightName)==list: subWeightName = subWeightName[0]
                            param = np.asarray(modelParams[layerName][weightName][subWeightName])
                            netParams += [param]
                    else:
                        param = np.asarray(modelParams[layerName][weightName])
                        netParams += [param]

            return graphInfo, [ NetParam('NP', p) for p in netParams ], trainInfo, trainState

        if fileName[-4:].lower() == '.npz':
            rootDic = np.load(fileName)
            paramNames = rootDic['ParamNameList']
            graphInfo['layersInfo'] = bytes2Str(rootDic['LayersInfo'].item())
            if 'blocks' in rootDic: graphInfo['blocks'] = [bytes2Str(x) for x in rootDic['blocks']]
            netParams = []
            for paramName in paramNames:
                netParams += [ rootDic[ bytes2Str(paramName) ] ]
            return graphInfo, [ NetParam('NP', p) for p in netParams ], trainInfo, trainState

        if (fileName[-4:].lower() == '.fbm') or (fileName[-5:].lower() == '.fbmc'):
            try:
                # Using Fireball's own FNJ (Fireball-Numpy-JSON) format.
                rootDic = loadFNJ(fileName)
                graphInfo = rootDic['graphInfo']
                netParams = NetParam.loadNetParams(rootDic)
                if 'trainInfo' in rootDic:      trainInfo = rootDic['trainInfo']
                if 'trainState' in rootDic:     trainState = rootDic['trainState']
            except:
                # Older version (Need to have a new version of layersInfo text to work.
                # This support is only for importing the network parameters. "trainState"
                # and "trainInfo" are not backward compatible and are ignored here.
                try:
                    # Old version of fbm running on old version of numpy
                    rootDic = np.load(fileName, encoding='latin1')
                    graphInfo = rootDic['graphInfo'].item()
                    netParams = NetParam.loadNetParams(rootDic)
                    # Note: trainState and trainInfo are not not backward compatible.
                    # if 'trainInfo' in rootDic:      trainInfo = rootDic['trainInfo'].item()
                    # if 'trainState' in rootDic:     trainState = rootDic['trainState'].item()
                    print('\nWarning: Old version of fbm file!')
                except:
                    # Old version of fbm running in new version of numpy => We need to use allow_pickle=True
                    print('\nWarning: Old version of fbm file (using allow_pickle)!')
                    rootDic = np.load(fileName, allow_pickle=True, encoding='latin1')
                    graphInfo = rootDic['graphInfo'].item()
                    netParams = NetParam.loadNetParams(rootDic)
                    # Note: trainState and trainInfo are not not backward compatible.
                    # if 'trainInfo' in rootDic:      trainInfo = rootDic['trainInfo'].item()
                    # if 'trainState' in rootDic:     trainState = rootDic['trainState'].item()

            return graphInfo, netParams, trainInfo, trainState
    
        raise RuntimeError("Don't know how to handle the file \"%s\"."%(fileName))

    # ******************************************************************************************************************
    def save(self, fileName, layersStr=None, blocks=None, netParams=None, epochInfo=None):
        r"""
        This function packages all the information in current network and uses the "saveFbm" class method to save it to the file specified by "fileName".
        
        Parameters
        ----------
        fileName : str
            The name of file containing network information. An 'fbm' extension is appended to the file name if it is not already included.
        
        layersStr : str
            A string containing the layers information. See the "Layers" class for more information about the format of the layersStr. If this is None, this function calls the "self.layers.getLayersStr" function to get the layers information from the current model.
            
        blocks : list
            A list of text strings or Block instances each defining a block. For more information about how blocks work, please refer to the "Block" and "BlockInstance" classes.
            
        netParams : list
            A list of NetParam objects containing the network parameters (weights and biases). If this is None, this function retrieves the network parameters from current model.
            
        epochInfo : tuple
            If not None, it is a tuple containing the following training state information:
            
                * epoch:          The last Epoch number in the previous training session.
                * batch:          The last batch number in the previous training session.
                * learningRate:   The value of learningRate in the last batch of last epoch in the previous training session
                * loss:           The value of loss in the last batch of last epoch in the previous training session
                * validMetric:    The validation ErrorRate/Accuracy/MSE/mAP calculated after the end the last epoch in the previous training session.
                * testMetric:     The test ErrorRate/Accuracy/MSE/mAP calculated after the end the last epoch in the previous training session.
                
            This information is saved to the file in the form of a dictionary.
            If this is None, the training state is not saved to the file.
        """
        if layersStr is None:   layersStr = self.layers.getLayersStr()
        if blocks is None:      blocks = self.layers.blocks
        if netParams is None:                               netParams = self.getAllNpNetParams()
        elif netParams[0].__class__.__name__ != 'NetParam': netParams = [ NetParam('NP', p) for p in netParams ]

        graphInfo = {
                        'name': self.name,
                        'producer': 'Fireball',
                        'version': fireball.__version__,
                        'doc': '',
                        'layersInfo': layersStr,
                        'blocks': [b if type(b)==str else b.getBlockStr() for b in blocks],
                    }
        trainInfo = {
                        'batchSize': self.batchSize,
                        'numEpochs': self.numEpochs,
                        'regFactor': self.regFactor,
                        'dropOutKeep': 1.0 - self.dropRate,
                        'learningRate': self.learningRate,
                        'optimizer': self.optimizer,
                        'savePeriod': self.savePeriod,
                        'saveBest': self.saveBest
                    }
        if self.lossFunction is not None:
            import inspect
            funcStr = inspect.getsource(self.lossFunction)
            funcStr= "def SAVED_LOSS_FUNCTION" + funcStr[funcStr.index("("):]
            trainInfo['lossFunction'] = funcStr
            
        trainState = None
        if epochInfo is not None:
            epoch, batch, learningRate, loss, validMetric, testMetric = epochInfo
            trainState = {
                            'epoch': epoch,
                            'batch': batch,
                            'learningRate': learningRate,
                            'loss': loss,
                            'testMetric': testMetric,
                            'validMetric': validMetric,
                            'bestEpoch': self.bestEpochInfo
                         }

        Model.saveFbm( graphInfo, trainInfo, netParams, trainState, fileName)

    # ******************************************************************************************************************
    @classmethod
    def saveFbm(cls, graphInfo, trainInfo, netParams, trainState, fbmFileName):
        r"""
        This function saves the given information to an "fbm" file.
        
        Parameters
        ----------
        graphInfo : dict
            A dictionary containing information about the network structure.
            
        trainInfo : dict
            A dictionary containing information about training the network (Num. of Epochs, BatchSize, etc)
            
        netParams : dict
            A list of NetParam objects containing the network parameter values (weights and biases).
            
        trainState : dict
            A dictionary containing the network state in the last training session. This is used to resume a training session if it is interrupted.
            
        fbmFileName : str
            The file name used to save the model information.
        """
        if fbmFileName[-4:].lower() != '.fbm':    fbmFileName += '.fbm'
        # Now using Fireball's own FNJ (Fireball-Numpy-JSON) format.
        rootDic = { 'graphInfo': graphInfo }
        if trainInfo is not None:   rootDic['trainInfo'] = trainInfo
        if trainState is not None:  rootDic['trainState'] = trainState
        NetParam.saveNetParams(rootDic, netParams)
        path = os.path.dirname(fbmFileName)
        if not os.path.exists(path):    os.makedirs(path)
        saveFNJ(fbmFileName, rootDic)

    # ******************************************************************************************************************
    def exportToOnnx(self, onnxFilePath, **kwargs):
        r"""
        This function exports current network information into an "ONNX" file. Please refer to the "fb2onnx.py" file for more information.
        
        Parameters
        ----------
        onnxFilePath : str
            The file name used to export the model information.

        **kwargs : dict
            A set of additional arguments passed directly to the "export" function of the "OnnxBuilder" class defined in the "fb2onnx" module. Here is a list of the arguments that may be included:
            
            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
                
            * **runQuantized (Boolean)**: True means include the codebooks and lookup functionality in the exported model so that the quantized model is executed at the inference time. This makes the exported model smaller at the expense of slightly increased execution time during the inference. If this is False for a quantized model, Fireball de-quantizes all parameters and includes the de-quantized information in the exported model. If this model is not quantized, then this argument is ignored.
            
            * **classNames (list of strings)**: If present, it must contains a list of class names for a classification model. The class names are then included in the exported model so that at the inference time the actual labels can easily be returned. If this is not present, then the class names are not included in the exported model and the inference code needs to convert predicted classes to the actual labels by some other means.
                
            * **graphDocStr (str)**: A text string containing documentation for the graph in the exported model. If present, this will be included in the exported onnx file.

            * **modelDocStr (str)**: A text string containing documentation for the exported model. If present, this will be included in the exported onnx file.

        """
        from . import fb2onnx
        fb2onnx.OnnxBuilder(self).export(onnxFilePath, **kwargs)
 
    # ******************************************************************************************************************
    def exportToTf(self, tfPath, **kwargs):
        r"""
        This function exports current network information to the specified directory. Please refer to the "fb2tf.py" file for more information.
        
        Parameters
        ----------
        tfPath : str
            The path to a folder that will contain the exported tensorflow files.

        **kwargs : dict
            A set of additional arguments passed directly to the "export" function
            defined in the "fb2tf" module. Here a list of the arguments that may
            be included:
            
            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
                
            * **runQuantized (Boolean)**: True means include the codebooks and lookup functionality in the exported model so that the quantized model is executed at the inference time. This makes the exported model smaller at the expense of slightly increased execution time during the inference. If this is False for a quantized model, Fireball de-quantizes all parameters and includes the de-quantized information in the exported model. If this model is not quantized, then this argument is ignored.
            
            * **classNames (list of str)**: If present, it must contains a list of class names for a classification model. The class names are then included in the exported model so that at the inference time the actual labels can easily be returned. If this is not present, then the class names are not included in the exported model and the inference code needs to convert predicted classes to the actual labels by some other means.
        """
        from . import fb2tf
        fb2tf.TfBuilder(self).export(tfPath, **kwargs)

    # ******************************************************************************************************************
    def exportToCoreMl(self, fileName, **kwargs):
        r"""
        This function exports current network information into a "CoreML" file. Please refer to the "fb2coreml.py" file for more information.
        
        Parameters
        ----------
        fileName : str
            The file name used to export the model information.
        
        **kwargs : dict
            A set of additional arguments passed directly to the "export" function
            of CmlBuilder class defined in the "fb2coreml" module. Here a list of
            the arguments that may be included:
            
            * **classNames (list)**: The class names used by the CoreML model. This is only used for classification problems.
        
            * **isBgr (Boolean)**: True means the images at the input of the model are in BGR format. This is used only for models that take an image as input.
            
            * **rgbBias (list or float)**: If this is a list, then it should contain the bias values for Red, green, and blue components (In the same order). If it is a float, it is used as bias for all 3 components. Also, this is used for the case of monochrome images. This is used only for models that take an image as input.
            
            * **scale (float)**: This is the scale that is applied to the input image before adding the rgbBias value(s) above. Basically, the "processedImage" which is actually fed to the model is defined as::
                
                    processedImage = scale x image + rgbBias
            
            * **maxSeqLen (int)**: The max sequence length used for NLP models. This defaults to 384 and used only if a sequence of token Ids are fed to the model in different NLP tasks.
                
            * **author (str)**: The text string used in the CodeML model for the author of the model.

            * **modelDesc (str)**: The text string used in the CodeML model as a short description of the model.

            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
                
        """
        from . import fb2coreml
        fb2coreml.CmlBuilder(self).export(fileName, **kwargs)

    # ******************************************************************************************************************
    def exportParamsNpz(self, fileName, orgNamesFile=None):
        r"""
        This function exports the network parameter to a numpy NZP file.
        
        Parameters
        ----------
        fileName : str
            The (NPZ) file name used to save the model information.
        
        orgNamesFile : str
            If specified, it must contain the path to the yaml file that was used to import the model from the original h5 file. In this case the names of parameter tensors are imported from this yaml file and used in the exported NPZ file.
        """
        netParams = self.getLayerParams(orgNamesFile=orgNamesFile)
        rootDic = dict( netParams )
        rootDic['ParamNameList'] = [ netParam[0] for netParam in netParams ]    # This gives the order of parameters
        rootDic['LayersInfo'] = self.layers.getLayersStr()
        if len(self.layers.blocks)>0:  rootDic['blocks'] = [b.getBlockStr() for b in self.layers.blocks]
        np.savez_compressed(fileName, **rootDic)

    # ******************************************************************************************************************
    @classmethod
    def makeFromFile(cls, modelPath=None, layersInfo=None,
                     trainDs=None, testDs=None, validationDs=None,
                     batchSize=None,
                     numEpochs=None,
                     regFactor=None,
                     dropOutKeep=None,
                     learningRate=None,
                     optimizer=None,
                     lossFunction=None,
                     name=None,
                     blocks=[],
                     saveModelFileName=None,
                     savePeriod=None,
                     saveBest=None,
                     gpus=None,
                     initParams=True):
        r"""
        This class method reads the network information from a file and creates a "Model" instance. This function uses the class method "loadModelFrom" to read the information.
        
        Parameters
        ----------
        modelPath : str
            The name of the file containing the network information. See the "loadModelFrom" function description for more information about the supported file formats.
            
        initParams : Boolean
            If True, the network parameters are initialized by the values in the specified model file. Otherwise, the network parameters are initialized randomly (Training from scratch).
            
        Others : The rest of parameters
            The rest of arguments are used to override the corresponding information read from the file. For a description of each argument please refer to the documentation of the :py:meth:`__init__` function above.
        
        Returns
        -------
        Model
            An instance of :py:class:`Model` class created from the information in the specified file.
            
        Note
        ----
        * If numEpochs is specified, this means we want a new training. In this case the training state loaded from the file is ignored.
        * If "saveModelFileName" is given and it exists, the model is loaded from this file. This is the case when a retraining of the original model was interrupted for any reason and we are now resuming it.
        """
        # We try to load from the saved model file first if it exists. This is the case when
        # a retraining of the original model was interrupted and we are resuming it.
        loadedModelPath = None
        if saveModelFileName is not None:
            if os.path.exists(saveModelFileName) and (os.stat(saveModelFileName).st_size>0):
                loadedModelPath = saveModelFileName

        if loadedModelPath is None:
            loadedModelPath = modelPath

        Model.printMsg('\nReading from "%s" ... '%(loadedModelPath), False)
        graphInfo, netParams, trainInfo, trainState = Model.loadModelFrom(loadedModelPath)
        
        Model.printMsg('Done.')
        
        modelName = graphInfo['name'] if name is None else name
        Model.printMsg('Creating the fireball model "%s" ... '%(modelName), False)

        if graphInfo is None:
            if os.path.exists(loadedModelPath)==False:
                raise ValueError("Could not find the specified model file \"%s\"!"%(loadedModelPath))
            raise RuntimeError("Could not find Model Configuration Information in the file \"%s\"."%(loadedModelPath))
        
        if trainInfo.get('lossFunction',None) is None:  SAVED_LOSS_FUNCTION = None
        else:
            ldict = {}
            exec(trainInfo['lossFunction'],globals(),ldict)
            SAVED_LOSS_FUNCTION = ldict['SAVED_LOSS_FUNCTION']

        blockStrs = graphInfo.get('blocks', [])
        blocks += [Block(s) for s in blockStrs]
        model = Model(modelName,
                      graphInfo['layersInfo'] if layersInfo is None else layersInfo,
                      trainDs, testDs, validationDs,
                      trainInfo['batchSize'] if batchSize is None else batchSize,
                      trainInfo['numEpochs'] if numEpochs is None else numEpochs,
                      trainInfo['regFactor'] if regFactor is None else regFactor,
                      trainInfo['dropOutKeep'] if dropOutKeep is None else dropOutKeep,
                      trainInfo.get('learningRate',None) if learningRate is None else learningRate,
                      trainInfo['optimizer'] if optimizer is None else optimizer,
                      SAVED_LOSS_FUNCTION if lossFunction is None else lossFunction,
                      blocks,
                      modelPath,
                      saveModelFileName,
                      trainInfo['savePeriod'] if savePeriod is None else savePeriod,
                      trainInfo['saveBest'] if saveBest is None else saveBest,
                      netParams=netParams if initParams else None,
                      trainingState=trainState if numEpochs is None else None,  # Ignore the loaded training state if
                                                                                # numEpoch is specified.
                      gpus=gpus)

        Model.printMsg('Done.')
        return model

    # ******************************************************************************************************************
    def createLrModel(self, modelPath, lrParams, **kwargs):
        r"""
        This function converts the specified parameters of this model to Low-Rank tensors based on the information in ``lrParams`` and then saves the resulting model to a file specified by ``modelPath``.
        
        Parameters
        ----------
        modelPath : str
            The path to the file that contains the model information for the converted model.
          
        lrParams : list
            This contains a list of tuples. The first element in each tuple is a layer name that specifies the layer to modify.
            
            The second parameter is the upper bound of the MSE between the original tensor and it's low-rank equivalent. The best "rank" value is found using this MSE value.
            
        **kwargs : dict
            A set of additional arguments. Here is a list of arguments that are currently supported:
            
            * **decomposeDWCN (Boolean)**: If false, the depth-wise convolutional layers are skipped. Otherwise, (the default), they are decomposed if specified in the lrParams.

            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
             
        Returns
        -------
        int
            Total number of parameters in the model after applying low-rank decomposition.
        """

        quiet = kwargs.get('quiet', False)
        decomposeDWCN = kwargs.get('decomposeDWCN', True)
        
        newLayersStr = ''
        newParams = []
        prevStage = 0
        newBlocks = []
        newNumParams = 0
        for layer in self.layers:
            layerMseUB = None
            for l,(layerScope, mseUB) in enumerate(lrParams):
                if layer.scope != layerScope:   continue
                layerMseUB = mseUB
            if layerMseUB is None:
                layerStr = layer.getLayerStr()
                layerParams = layer.netParamValues
                newParams += layerParams
                newNumParams += sum(x.size for x in layerParams)
            else:
                newLayerParams, decLayerStr, newNumLayerParams, infoStr = layer.decompose(self.session,
                                                                                          ('lr', layerMseUB, None),
                                                                                          decomposeDWCN)
                if not quiet:   print("  " + infoStr)
                if newLayerParams is None:
                    layerStr = layer.getLayerStr()
                    layerParams = layer.netParamValues
                    newParams += layerParams
                    newNumParams += sum(x.size for x in layerParams)
                elif layer.__class__.__name__ == 'BlockInstance':
                    blockName = '%sc%d'%(layer.name,len(newBlocks)+1)
                    newBlocks += [ Block(blockName + '|' + decLayerStr) ]
                    layerStrs = [ blockName ]
                    if (len(layer.postActivations) > 0) or (layer.activation != 'none'):
                        layerStrs += [layer.activation]
                        if len(layer.postActivations) > 0:
                            layerStrs += [ pa.getLayerStr() for pa in layer.postActivations ]
                    layerStr = ':'.join(layerStrs)
                    newParams += newLayerParams
                    newNumParams += newNumLayerParams
                else:
                    layerStr = decLayerStr
                    newParams += newLayerParams
                    newNumParams += newNumLayerParams

            if layer.stage != prevStage:    newLayersStr += ';' + layerStr
            else:                           newLayersStr += ',' + layerStr
            prevStage = layer.stage
        
        newLayersStr = newLayersStr[1:]
        self.save(modelPath, newLayersStr, self.layers.blocks+newBlocks, newParams )
        if not quiet:   print('Total New Parameters: {:,}'.format(newNumParams))
        return newNumParams

    # ******************************************************************************************************************
    def createPrunedModel(self, modelPath, prnParams, **kwargs):
        r"""
        This function reduces the number of non-zero parameters by pruning the ones close to zero. The pruning is applied to the layers specified in the ``prnParams``. The resulting model is saved to the file specified by ``modelPath``.
        
        Parameters
        ----------
        modelPath : str
            The path to the file that contains the model information for the converted (pruned) model.
          
        prnParams : list
            This contains a list of tuples. The first element in each tuple is a layer name that specifies the layer to modify.
            
            The second parameter is the upper bound of the MSE between the original tensor and it's pruned version.
            
        **kwargs : dict
            A set of additional arguments. Here is a list of arguments that are currently supported:
            
            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
             
        Returns
        -------
        int
            Total size of non-zero parameters in bytes.
        """
        quiet = kwargs.get('quiet', False)
        
        newNetParams = []
        newBlocks = []
        newNumParams = 0
        prunedBytes = 0
        for layer in self.layers:
            layerMseUB = None
            for l,(layerScope, mseUB) in enumerate(prnParams):
                if layer.scope != layerScope:   continue
                layerMseUB = mseUB
            if layerMseUB is None:
                newNetParams += layer.getNpParams()
                prunedBytes += layer.getNumBytes()
            else:
                newLayerNetParams, layerPrunedBytes, infoStr = layer.prune(layerMseUB)
                if not quiet:   print("  " + infoStr)
                newNetParams += newLayerNetParams
                prunedBytes += layerPrunedBytes

        self.save(modelPath, netParams=newNetParams )
        if not quiet:   print('Total New Parameters Size (bytes): {:,}'.format(prunedBytes))
        return prunedBytes
    
    # ******************************************************************************************************************
    @classmethod
    def pruneModel(cls, inputModelPath, outputModelPath, mseUb, **kwargs):
        r"""
        This class method reads the model information from "inputModelPath", prunes its parameters based on the ``mseUb`` and ``minReductionPercent`` parameters, and saves the new pruned model to "outputModelPath".
        
        Parameters
        ----------
        inputModelPath : str
            The path to the input file that is about to be pruned.

        outputModelPath : str
            The path to the resulting quantized file.

        mseUb : float
            The Upper bound for the MSE between the original and pruned parameters.

        **kwargs : dict
            A set of additional arguments passed to the downstream functions. Here is a list of the arguments that may be included:
            
            * **minReductionPercent (int)**: If provided, it specifies the minimum percentage of reduction required for each tensor. If this percentage cannot be achieved, the tensor is not pruned.

            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
                
            * **verbose (Boolean)**: If not running in parallel (``numWorkers=0``), setting this to True causes a line to be printed with detailed results for each model parameter processes. Otherwise only the progress is displayed. This parameter is ignored if ``quiet`` is True. This also has no effect if running in parallel mode (``numWorkers>0``).

            * **numWorkers (int)**: The number of worker threads pruning in parallel. 0 means single thread operation which is slower but can be more verbose. None (default) lets fireball to decide this values based on the number of CPUs available.

        Returns
        -------
        totalPruned : int
            Total number of parameters pruned.
        sOut : int
            The size of new pruned model.
        """

        quiet = kwargs.get('quiet', False)
        kwargs['mseUb'] = mseUb     # Put mseUb in the kwargs to be used by downstream functions

        if not quiet:   myPrint('\nReading model parameters from "%s" ... '%(inputModelPath), False)
        graphInfo, netParams, trainInfo, _ = cls.loadModelFrom(inputModelPath)
        if not quiet:   myPrint('Done.')
                    
        t0 = time.time()
        
        newNetParams, orgNumParams, totalPruned = NetParam.pruneNetParams(netParams, **kwargs)

        if not quiet:
            myPrint('Pruning process complete (%.2f Sec.)'%(time.time()-t0))
            myPrint('Now saving to "%s" ... '%(outputModelPath), False)
            
        if outputModelPath[-4:].lower() != '.fbm':    outputModelPath += '.fbm'
        Model.saveFbm(graphInfo, trainInfo, newNetParams, None, outputModelPath)
        
        if not quiet:
            myPrint('Done.')
            myPrint('\nNumber of parameters: {:,} -> {:,} ({:,} pruned)'.format( orgNumParams,
                                                                                 orgNumParams-totalPruned,
                                                                                 totalPruned))
        sIn = os.stat(inputModelPath).st_size
        sOut = os.stat(outputModelPath).st_size
        
        if not quiet: myPrint('Model File Size: {:,} -> {:,} bytes'.format( sIn, sOut))
        
        return (totalPruned, sOut)
        
    # ******************************************************************************************************************
    @classmethod
    def quantizeModel(cls, inputModelPath, outputModelPath, mseUb, **kwargs):
        r"""
        This class method reads the model information from "inputModelPath", quantizes its parameters based on the information in "qInfo", and saves the new quantized model to "outputModelPath".
        
        Note
        ----
        This method is used for trained quantization. The results is not necessarily optimized for entropy coding.
                
        Parameters
        ----------
        inputModelPath : str
            The path to the input file that is about to be quantized.
            
        outputModelPath : str
            The path to the resulting quantized file.
            
        mseUb : float
            The Upper bound for the MSE between the original and quantized parameters.
                           
        **kwargs : dict
            A set of additional arguments passed to the downstream functions. Here is a list of the arguments that may be included:
            
            * **reuseEmptyClusters (Boolean)**: True (default) means keep reusing/reassigning empty clusters during the K-Means algorithm. In this case, the final number of clusters is a power of 2 (between ``minBits`` and ``maxBits``). False means we remove the empty clusters from the codebook and the final number of clusters may not be a power of 2. (between ``minSymCount`` and ``maxSymCount``)
            
            * **weightsOnly (Boolean)**: True (default) means quantize only weight matrixes. Biases and BatchNorm parameters are not quantized. False means quantize any network parameter if possible.

            * **minSymCount (int)**: The minimum number of symbols in the quantized tensors. Fireball does a binary search between ``minSymCount`` and ``maxSymCount`` to find the best symbol count that results in a quantization error (MSE) below the specified ``mseUb``. The found symbol count is used for the initial size of codebook. The default is 4. Ignored if ``reuseEmptyClusters`` is True.

            * **maxSymCount (int)**: The maximum number of symbols in the quantized tensors. Fireball does a binary search between ``minSymCount`` and ``maxSymCount`` to find the best symbol count that results in a quantization error (MSE) below the specified ``mseUb``. The found symbol count is used for the initial size of codebook. The default is 4096. Ignored if ``reuseEmptyClusters`` is True.

            * **minBits (int)**: The minimum number of quantization bits for the quantized tensors. Fireball searches for  the lowest quantization bits (``qBits``) between ``minBits`` and ``maxBits`` that results in a quantization error (MSE) below the specified ``mseUb``. The quantization bits found ``qBits`` defines the codebook size (codebookSize=symCount=2^qBits). The default is 2. Ignored if ``reuseEmptyClusters`` is False.

            * **maxBits (int)**: The maximum number of quantization bits for the quantized tensors. Fireball searches for  the lowest quantization bits (``qBits``) between ``minBits`` and ``maxBits`` that results in a quantization error (MSE) below the specified ``mseUb``. The found ``qBits`` value defines the codebook size (codebookSize=symCount=2^qBits). The default is 12. Ignored if ``reuseEmptyClusters`` is False.

            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
                
            * **verbose (Boolean)**: If not running in parallel (``numWorkers=0``), setting this to True causes a line to be printed with detailed results for each model parameter processes. Otherwise only the progress is displayed. This parameter is ignored if ``quiet`` is True. This also has no effect if running in parallel mode (``numWorkers>0``).

            * **numWorkers (int)**: The number of worker threads quantizing in parallel. 0 means single thread operation which is slower but can be more verbose. None (default) lets fireball to decide this values based on the number of CPUs available.

        Returns
        -------
        originalBytes : int
            The original size of parameter data in bytes.
        quantizedBytes : int
            The quantized size of parameter data in bytes.
        sOut : int
            The file size of new quantized model.
        """
        quiet = kwargs.get('quiet', False)
        weightsOnly = kwargs.get('weightsOnly', True)
        kwargs['mseUb'] = mseUb     # Put mseUb in the kwargs to be used by downstream functions
        
        if not quiet:   myPrint('\nReading model parameters from "%s" ... '%(inputModelPath), False)
        graphInfo, netParams, trainInfo, _ = cls.loadModelFrom(inputModelPath)
        if not quiet:   myPrint('Done.')
        
        t0 = time.time()
        quantizedNetParams, originalBytes, quantizedBytes = NetParam.quantizeNetParams(netParams, **kwargs)
                
        if not quiet:
            myPrint('Quantization complete (%.2f Sec.)'%(time.time()-t0))
            myPrint('Now saving to "%s" ... '%(outputModelPath), False)

        if outputModelPath[-4:].lower() != '.fbm':    outputModelPath += '.fbm'
        Model.saveFbm(graphInfo, trainInfo, quantizedNetParams, None, outputModelPath)
        if not quiet:
            myPrint('Done.')
            myPrint('\nSize of Data: {:,} -> {:,} bytes'.format( originalBytes, quantizedBytes))
        
        sIn = os.stat(inputModelPath).st_size
        sOut = os.stat(outputModelPath).st_size
        
        if not quiet:   myPrint('Model File Size: {:,} -> {:,} bytes'.format( sIn, sOut))
        
        return (originalBytes, quantizedBytes, sOut)

    # ******************************************************************************************************************
    @classmethod
    def compressModel(cls, inputModelPath, outputModelPath, **kwargs):
        r"""
        This class method reads the model information from ``inputModelPath``, compresses its parameters using arithmetic coding and saves the new quantized model to ``outputModelPath``.
                        
        Parameters
        ----------
        inputModelPath : str
            The path to the input file that is about to be compressed.
            
        outputModelPath : str
            The path to the resulting compressed file.
                                       
        **kwargs : dict
            A set of additional arguments passed to the downstream functions. Here is a list of the arguments that may be included:
            
            * **quiet (Boolean)**: True means there are no messages printed during the execution of the function.
            
            * **verbose (Boolean)**: If not running in parallel (``numWorkers=0``), setting this to True causes a line to be printed with detailed results for each model parameter processes. Otherwise only the progress is displayed. This parameter is ignored if ``quiet`` is True. This also has no effect if running in parallel mode (``numWorkers>0``).

            * **numWorkers (int)**: The number of worker threads compressing in parallel. 0 means single thread operation which is slower but can be more verbose. None (default) lets fireball to decide this values based on the number of CPUs available.
            
        Returns
        -------
        sIn : int
            The original file size.
        sOut : int
            The file size of new compressed model.
        """
        import fireball.arithcoder as ac

        quiet = kwargs.get('quiet', False)
        
        if not quiet:   myPrint('\nReading model parameters from "%s" ... '%(inputModelPath), False)
        graphInfo, netParams, trainInfo, trainState = cls.loadModelFrom(inputModelPath)
        if not quiet:   myPrint('Done.')

        t0 = time.time()
        netParamDics = NetParam.compressNetParams(netParams, **kwargs)

        if not quiet:
            myPrint('Finished compressing model parameters (%.2f Sec.)'%(time.time()-t0))
            myPrint('Now saving to "%s" ... '%(outputModelPath), False)

        if outputModelPath[-5:].lower() != '.fbmc':    outputModelPath += '.fbmc'

        rootDic = { 'graphInfo': graphInfo }
        if trainInfo is not None:   rootDic['trainInfo'] = trainInfo
        if trainState is not None:  rootDic['trainState'] = trainState
        rootDic['netParamDics'] = netParamDics
        
        path = os.path.dirname(outputModelPath)
        if not os.path.exists(path):    os.makedirs(path)
        saveFNJ(outputModelPath, rootDic, flags=1)

        sIn = os.stat(inputModelPath).st_size
        sOut = os.stat(outputModelPath).st_size

        if not quiet:
            myPrint('Done.')
            myPrint('Model File Size: {:,} -> {:,} bytes'.format( sIn, sOut))
        
        return (sIn, sOut)

    # ******************************************************************************************************************
    def createLdrModel(self, modelPath, ldrParams):
        r"""
        This function converts the specified fully connected layers of this model to LDR layers and saves the resulting model to a file specified by ``modelPath``.
        
        Parameters
        ----------
        modelPath : str
            The path to the file that will contain the model information for the converted model.
        
        ldrParams: list
            This contains a list of tuples. Each tuple has 4 parameters as follows:
            
            * **layer Index**: Specifying the layer that should be modified.
            * **e**: The "e" value used in LDR process.
            * **f**: The "f" value used in LDR process.
            * **rank**: The displacement rank used in the LDR process.
                
            **Note:** The layer indexes of the tuples in the list should be
            in increasing order.
        """
        # ldrParams: [(l1,e1,f1,r1), (l2,e2,f2,r2),...] where l1<l2<...
        netParams = self.session.run(self.towers[0].tfNetParams)
        newNetParams = []
        newLayersInfo = []
        p = 0       # netParams counter
        ldri = 0    # Current ldr parameter index
        for l,layerInfo in enumerate(self.layersInfo):
            layerType = layerInfo['type']
            if ldri<len(ldrParams):
                if l == ldrParams[ldri][0]:
                    if layerType=='fc':
                        weights = netParams[p]
                        biases = netParams[p+1]
                        p+=2

                        e,f,r = ldrParams[ldri][1],ldrParams[ldri][2],ldrParams[ldri][3]
                        g,jH,e,f,r,mse = ldr.getLdrGH(weights, e, f, r, None, np.float32)
                        self.printMsg('    %s Layer %d, e,f: %d,%d, Rank %d, MSE= %f'%(layerType.upper(),l,e,f,r,mse))
                        
                        newLayerInfo = layerInfo.copy()
                        newLayerInfo['type'] = 'ldr'
                        newLayerInfo['rank'] = r
                        newLayerInfo['e'] = e
                        newLayerInfo['f'] = f
                        newLayersInfo += [ newLayerInfo ]

                        newNetParams += [g, jH, biases]
                        ldri += 1
                        continue

                    self.printMsg('Layer %d is not Fully Connected. Ignoring!'%(l))

            newLayersInfo += [layerInfo]
            newNetParams += [netParams[p], netParams[p+1]]
            p += 2
            if layerType in ['lr','ldr']:
                newNetParams += [netParams[p]]
                p += 1

        self.save(modelPath, newLayersInfo, newNetParams)
    
    # ******************************************************************************************************************
    @classmethod
    def resetGraph(cls):
        r"""
        A utility class method that resets the default graph of TensorFlow.
        """
        tf.reset_default_graph()

    # ******************************************************************************************************************
    def getMeanGrads(self, towerGrads):
        r"""
        This function receives the gradient information from different towers and calculates the average gradient value for each variable in the network. It then returns the results in a list of tuples. Each tuple in the list has the mean gradient of a variable and the variable itself.
        
        Parameters
        ----------
        towerGrads: list
            towerGrads is a list of lists. Each list is related to a single tower and contains tuples of gradients and variables for all network parameters taken from that tower. See the format in the following comments.
        
        Returns
        -------
        list
            A list of tuples where each tuple in the list has the mean gradient of a variable and the variable itself.
        """
        # towerGrads is a list containing grads and vars pairs for all variables on all Towers:
        # towerGrads = [ [ (T0Grad0, T0Var0), (T0Grad1, T0Var1), ... (T0GradN,T0VarN) ],    > Grads and vars for Tower0
        #                [ (T1Grad0, T1Var0), (T1Grad1, T1Var1), ... (T1GradN,T1VarN) ],    > Grads and vars for Tower1
        #                ...
        #                [ (TMGrad0, TMVar0), (TMGrad1, TMVar1), ... (TMGradN,TMVarN) ] ]   > Grads and vars for TowerM
        meanGradsAndVars = []
        with tf.name_scope('MeanGrads'):
            for gradVarsForGpus in zip(*towerGrads):
                # Each gradVarsForGpus is tuple of (grad,var) pairs one per GPU for the same variable. For example
                # gradVarsForGpus for the first iteration would be:
                #    ( (T0Grad0, T0Var0), (T1Grad0, T0Var0), ... (TMGrad0, T0Var0) )
                gradsList = []
                for g, v in gradVarsForGpus:
                    # v and g in i'th iteration are the variable and its gradient on the i'th tower.
                    # Note that g is a tensor the same shape as the variable v. We want to create a list of g's, which
                    # can be represented with a tensor that has one dimension more than the original g.
                    if g is None: continue
                    gradsList += [ tf.expand_dims(g, 0) ]
                if len(gradsList) == 0: continue
                
                grads = tf.concat(axis=0, values=gradsList)
                # grads is now a tensor that has one dimension more than g's and grads[i] is the gradient values for
                # this variable on i'th tower.

                # Average over dimension 0 (the 'tower' dimension).
                meanGrad = tf.reduce_mean(grads, 0)
                var = gradVarsForGpus[0][1]             # Variables are shared. Just use the one from the first tower
                meanGradsAndVars += [ (meanGrad, var) ]
                self.tfSummaries += [ tf1.summary.histogram(var.op.name + '/meanGrad', meanGrad) ]

        return meanGradsAndVars

    # ******************************************************************************************************************
    def updateBnMovingAverages(self, batchNormMoments):
        r"""
        This function receives the batch normalization moving averages information from different towers and creates a list of operators to update the aggregated moving average values for the whole model. It then returns the list of operators.
        
        Parameters
        ----------
        batchNormMoments: list
            batchNormMoments is a list containing bnValues, bnVars pairs for all BN variables on all Towers.
            
        Returns
        -------
        list
            A list of operators used to update the moving averages for the batch normalization layers.
        """
        # batchNormMoments is a list containing bnValues, bnVars pairs for all BN variables on all Towers:
        # towerGrads = [ [ (T0BnValues0, T0BnVar0), (T0BnValues1, T0BnVar1), ... (T0BnValuesN,T0BnVarN) ],    > bnValues and bnVars for Tower0
        #                [ (T1BnValues0, T1BnVar0), (T1BnValues1, T1BnVar1), ... (T1BnValuesN,T1BnVarN) ],    > bnValues and bnVars for Tower1
        #                ...
        #                [ (TMBnValues0, TMBnVar0), (TMBnValues1, TMBnVar1), ... (TMBnValuesN,TMBnVarN) ] ]   > bnValues and bnVars for TowerM
        decay = 0.99
        updateBnOps = []
        with tf.name_scope('UpdateBNMovingAverages'):
            for momentVarsForGpus in zip(*batchNormMoments):
                # Each momentVarsForGpus is tuple of (moment,var) pairs one per GPU for the same variable. For example
                # momentVarsForGpus for the first iteration would be:
                #    ( (T0BnValues0, T0BnVar0), (T1BnValues0, T1BnVar0), ... (TMBnValues0, TMBnVar0) )
                bnValuesList = []
                for bnVal, bnVar in momentVarsForGpus:
                    # bnVal and bnVar in i'th iteration are the bnValue for the bnVar variable on the i'th tower.
                    # Note that bnVal is a tensor the same shape as the variable bnVar. We want to create a list of
                    # bnVal's, which can be represented with a tensor that has one dimension more than the original
                    # bnVal.
                    bnValuesList += [ tf.expand_dims(bnVal, 0) ]

                bnValues = tf.concat(axis=0, values=bnValuesList)
                # bnValues is now a tensor that has one dimension more than bnVal's and bnValues[i] is the BN values for
                # this variable on i'th tower.

                # Average over dimension 0 (the 'tower' dimension).
                meanBnValues = tf.reduce_mean(bnValues, 0)
                bnVar = momentVarsForGpus[0][1]     # BN Variables are shared. Just use the one from the first tower
                self.tfSummaries += [ tf1.summary.histogram(bnVar.op.name + '/mean', meanBnValues) ]

                updateBnOps += [ tf1.assign(bnVar, bnVar*decay + meanBnValues*(1.0-decay)) ]

        return updateBnOps

    # ******************************************************************************************************************
    def buildTowers(self, initValues):
        r"""
        This function creates the TensorFlow graph on multiple towers. Each tower corresponds to a single GPU device. If there are no GPU cards on the current machine, then only one tower is created that will run on the CPU.
        Fireball first creates the "Tower" instances and have them create the network graphs. It then merges the outputs of the tower graphs to create the outputs of the whole model. The whole TensorFlow graph for this model is created in this function.
        
        Parameters
        ----------
        initValues : list
            initValues is a list of NetParam objects that are used to initialize the network parameters. This is for the case where this network is being created from a pre-trained network file. If this is None, all network variables are initialized based on their specified initializers (random values or constants).
        """
        self.tfSummaries = []
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                # Create the optimizer and placeholders on CPU:
                with tf.name_scope('LearningRateDecay'):
                    # Setting up learning rate decay:
                    # Decay once per epoch, using an exponential schedule starting at self.learningRateInit.
                    self.tfBatch = tf.Variable(0, name='BatchCounter', trainable=False)     # Incremented once per batch (This is our global step)
                    warmUpLr = tf.cast(self.tfBatch, tf.float32)*self.learningRateInit/(self.learningRateWarmUp+0.1)
                    if self.learningRatePieces is not None:
                        lrBoundaries, lrValues = self.learningRatePieces
                        # Fixing Boundaries: Ignore the first one, decrement to match expected schedule
                        lrBoundaries = [x-1 for x in lrBoundaries[1:]]
                        self.tfLearningRate = tf.cond(self.tfBatch < self.learningRateWarmUp, lambda: warmUpLr,
                                                      lambda: tf1.train.piecewise_constant(self.tfBatch,
                                                                                           lrBoundaries, lrValues,
                                                                                           name='LearningRate'))
                    elif self.lrDecayStep == 0:
                        self.tfLearningRate = tf1.constant(self.learningRateInit, dtype=tf.float32, name='LearningRate')
                    else:
                        batchWU = self.tfBatch - self.learningRateWarmUp
                        self.tfLearningRate = tf.cond(self.tfBatch < self.learningRateWarmUp, lambda: warmUpLr,
                                                      lambda: tf1.train.exponential_decay(self.learningRateInit,# Base learning rate
                                                                                          batchWU,              # Batch Number
                                                                                          self.lrDecayStep,     # Decay step.
                                                                                          LEARNING_DECAY_RATE,  # Decay rate.
                                                                                          staircase=True,
                                                                                          name='LearningRate'))
                    
                    self.tfSummaries += [ tf1.summary.scalar('LearningRate', self.tfLearningRate) ]

                if self.optimizer == 'GradientDescent':
                    self.tfOptimizer = tf.train.GradientDescentOptimizer(self.tfLearningRate)

                elif self.optimizer == 'Momentum':
                    self.tfOptimizer = tf1.train.MomentumOptimizer(self.tfLearningRate, 0.9)
        
                else:   # 'auto', 'Adam'
                    self.tfOptimizer = tf1.train.AdamOptimizer(self.tfLearningRate)

            if self.gpus[0]==-1:    self.towers = [ Tower(self, initValues=initValues) ]   # No GPUs
            else:   self.towers = [ Tower(self, t, '/gpu:%d'%(t), initValues) for t,gpu in enumerate(self.gpus) ]
            towerGrads = [ tower.getGrads(self.tfOptimizer) for tower in self.towers ]
            towerLosses = [ tower.tfLoss for tower in self.towers ]
            towerInferPreds = [ tower.tfInferPrediction for tower in self.towers ]
            towerEvalResults = [ tower.evalResults for tower in self.towers ]
            towerAbsGrads = [ tower.tfAbsGrads for tower in self.towers ]
            batchNormMoments = [ tower.batchNormMoments for tower in self.towers ]
            
            if not self.quiet:
                if (initValues is not None) and (len(self.towers[0].tfNetParams)>len(initValues)):
                    myPrint("\nWarning: Only %d of %d parameter tensors initialized from the specified " \
                            "file!"%(len(initValues), len(self.towers[0].tfNetParams)), color='yellow')

            with tf.device('/cpu:0'):
                # Average the gradients from all towers
                allGrads, ntGrads = [list(g) for g in zip(*towerGrads)]
#                meanGradsAndVars = self.getMeanGrads(allGrads)
                meanGradsAndVars = None if allGrads[0] is None else self.getMeanGrads(allGrads)
                meanGradsAndVarsNT = None if ntGrads[0] is None else self.getMeanGrads(ntGrads)

                with tf.name_scope('Loss'):
                    # Average the Losses from all towers
                    tfLosses = tf.stack(axis=0, values=towerLosses)
                    self.tfLoss = tf.reduce_mean(tfLosses, 0, name='LossMean')
                    self.tfSummaries += [ tf1.summary.scalar('Loss', self.tfLoss) ]

                with tf.name_scope('SumAbsGrads'):
                    # Add the absolute values of Loss Gradients from all towers. Each tensor in "self.tfAbsGrads"
                    # will be the absolute values of gradient of loss with respect to a specific variable summed
                    # up for all towers.
                    paramTowerGrads = zip(*towerAbsGrads)
                    self.tfAbsGrads = [tf.add_n(towerGrads) for towerGrads in paramTowerGrads]


                if type(towerInferPreds[0])==tuple:
                    # towerInferPreds is a list of one tuple for each tower:
                    #       [(t0x0, t0x1, ...), (t1x0,t1x1,...), ...]
                    # we want to change this to:
                    #       (x0, x1,...)
                    # where xi is the concatenation of [t0xi, t1xi, ...]
                    tupLen = len(towerInferPreds[0])
                    self.tfInferPrediction = tuple( tf.concat(axis=0, values=[x[i] for x in towerInferPreds]) for i in range(tupLen) )
                else:
                    self.tfInferPrediction = tf.concat(axis=0, values=towerInferPreds, name='InferPrediction')
            
                if towerEvalResults[0] is None: self.evalResults = None
                else:
                    # evalResults is only used for the multi-dimensional regression cases. It contains Sum Squared Error
                    # and Sum Absolute Error for each sample.
                    self.evalResults = tuple( tf.concat(axis=0, values=[x[i] for x in towerEvalResults]) for i in range(2) )
                
                # Apply the gradients and update BatchNorm Moving Averages
                with tf.control_dependencies( self.updateBnMovingAverages(batchNormMoments) ):
                    self.tfOptimize = self.tfOptimizer.apply_gradients(meanGradsAndVars, global_step=self.tfBatch)
                    self.tfOptimizeNT = None
                    if (meanGradsAndVarsNT is not None) and (self.trainAllBatch>0):
                        self.tfOptimizeNT = self.tfOptimizer.apply_gradients(meanGradsAndVarsNT, global_step=self.tfBatch)

            self.tfSummaries = tf1.summary.merge(self.tfSummaries)
            self.initializer = tf1.global_variables_initializer()

    # ******************************************************************************************************************
    def initSession(self, session=None, summaryWriter=None):
        r"""
        Creates and initializes a TensorFlow session for this model.
        
        Parameters
        ----------
        session : TensorFlow session object
            If this is None, a new session is created and kept by this class for all TensorFlow operations. Otherwise, the specified session will be used by this class.
            
        summaryWriter: TensorFlow summaryWriter object, str, or Boolean
            * If this is an str, it contains the path to the TensorFlow summary information folder.
            * If this is Boolean and the value is True, a new path called "TensorBoard" is created and used to save the TensorFlow summary information.
            * If this is None, the summaryWriter is disabled.
            * Otherwise this should be a TensorFlow summaryWriter object.
        """
        if session is None:
            # If gpus==[-1], it means we don't want to use GPUs.
            # In this case we set visibleGpus = "0" which means GPU0 is visible. But it is not used
            # Couldn't make all GPUs invisible for this case.
            visibleGpus = "0" if self.gpus==[-1] else ','.join(str(x) for x in self.gpus if x>=0)
            tfConfig = tf1.ConfigProto(allow_soft_placement=True,        # Needed for proper operation of towers
                                       gpu_options=tf1.GPUOptions(allow_growth=True, visible_device_list=visibleGpus))
            
            self.session = tf1.Session(config=tfConfig, graph=self.graph)
            self.session.run( self.initializer )
        else:
            self.session = session

        summaryDir = None
        self.summaryWriter = None
        if type(summaryWriter) is str:      # Path to the summaryWriter file is given
            summaryDir = summaryWriter
        elif type(summaryWriter) is bool:   # Create a summaryWriter in <curPath>/Tensorboard
            if summaryWriter:
                summaryDir = os.path.dirname(sys.modules['__main__'].__file__) + 'TensorBoard'
        elif summaryWriter is not None:
            self.summaryWriter = summaryWriter

        if summaryDir is not None:
            self.summaryWriter = tf.summary.FileWriter(summaryDir, self.session.graph)
        
    # ******************************************************************************************************************
    def train(self, logBatchInfo=False):
        r"""
        Trains the network using the training data in "trainDs". The training stops after the specified number of epochs or if the minimum-loss criteria are met. The "maxLoss", "minLoss" and "minLossCount" parameters can be set before calling this function.
        
        This function contains the main training loop. It acquires the batches of training data iteratively and uses them to train the network. At the end of each epoch, this function calculates the test and/or validation errors if possible and prints one row in the training table.
        
        Parameters
        ----------
        logBatchInfo : Boolean
            If true, this function logs the learning rate and loss values for each batch during the training. After training this information is available in the ``batchLogInfo`` dictionary. This dictionary object has the following items:
            
            * loss: A 1-D array of floating point values specifying the training loss for each batch.
            * learningRate: a 1-D array of floating point values specifying the learning rate used for each batch of training datas.
        """
        assert self.trainDs is not None, "A training dataset in not available!"
        startTime = time.time()
        
        testMetric = None       # Metric can be error, accuracy, mAP, MSE, etc.
        validMetric = None      # Metric can be error, accuracy, mAP, MSE, etc.
        self.bestMetric = None
        self.bestEpochInfo = None
        self.sumAbsGrads = None     # Invalidate any previously calculated grads.
                        
        numMinLoss = 0
        epoch = 0
        if self.trainingState is not None:
            epoch = self.trainingState['epoch']
            batch = self.trainingState['batch']
            learningRate = self.trainingState['learningRate']
            loss = self.trainingState['loss']
            testMetric = self.trainingState.get('testMetric', None)
            validMetric = self.trainingState.get('validMetric', None)
            self.bestEpochInfo = self.trainingState['bestEpoch']

            if epoch > 0:
                if (epoch+1) >= self.numEpochs:
                    self.printMsg('The model has already been trained. Here is the last training results:')
                else:
                    self.printMsg('Resuming the interrupted training process (Last Epoch: %d) ...'%(epoch+1))
                    
                self.updateTrainingTable('start')
                if self.bestEpochInfo is not None:
                    e, b, lr, ls, tm, vm = self.bestEpochInfo
                    if vm is not None:      self.bestMetric = vm
                    elif tm is not None:    self.bestMetric = tm
                    if e != epoch:
                        self.updateTrainingTable('addrow', self.bestEpochInfo)
                    
                self.updateTrainingTable('addrow', (epoch, batch, learningRate, loss, validMetric, testMetric))
                if (epoch+1)<self.numEpochs:
                    self.updateTrainingTable('separator')
                    self.tfBatch.load(batch, self.session)
                epoch += 1
   
        if epoch==0:
            self.updateTrainingTable('start')

        numTowers = len(self.towers)
        assert ((self.batchSize % numTowers)==0), 'BatchSize must be a multiple of number of towers!'

        # Make sure we can reproduce results. Note that this may not work as expected if training is interrupted.
        np.random.seed(1234)
        if logBatchInfo: self.batchLogInfo = { 'loss': [], 'learningRate': [] }
        batch = 0
        while epoch < self.numEpochs:
            sumLoss = 0
            for b, (batchSamples, batchLabels) in enumerate(self.trainDs.batches(self.batchSize)):
                self.updateTrainingTable('batch', (epoch+1, b+1))
                feedDic = self.layers.input.feed(batchSamples, self.towers)
                if feedDic is None: break   # We are done with this epoch
                feedDic.update( self.layers.output.feed(batchLabels, self.towers) )
                
                thingsToDo = [ self.tfOptimize if batch>=self.trainAllBatch else self.tfOptimizeNT ]
                thingsToDo += [self.tfLearningRate, self.tfBatch, self.tfLoss]
                if self.summaryWriter is not None:
                    thingsToDo += [ self.tfSummaries ]
                    _, learningRate, batch, loss, summaries = self.session.run(thingsToDo, feed_dict=feedDic)
                    self.summaryWriter.add_summary(summaries, batch)
                else:
                    _, learningRate, batch, loss = self.session.run(thingsToDo, feed_dict=feedDic)

                if self.afterBatch is not None:
                    self.afterBatch(epoch, b, learningRate, loss)
                
                if logBatchInfo:
                    self.batchLogInfo['learningRate'] += [ learningRate ]
                    self.batchLogInfo['loss'] += [ loss ]
                    
                sumLoss += loss

            # At the end of Epoch, calculate loss average and evaluate the model using
            # test and/or validation datasets
            loss = sumLoss/b
            testMetric = self.evaluateDSet(self.testDs, returnMetric=True)
            validMetric = self.evaluateDSet(self.validationDs, returnMetric=True)
            
            epochInfo = (epoch, int(batch), float(learningRate), float(loss), validMetric, testMetric)

            isBestEpoch = False
            if validMetric is not None:
                if self.bestMetric is None:             isBestEpoch = True
                elif self.validationDs.evalMetricBiggerIsBetter:
                    if validMetric > self.bestMetric:   isBestEpoch = True
                elif validMetric < self.bestMetric:     isBestEpoch = True
                if isBestEpoch:
                    self.bestMetric = validMetric
                    self.bestEpochInfo = epochInfo
                    
            elif testMetric is not None:
                if self.bestMetric is None:             isBestEpoch = True
                elif self.testDs.evalMetricBiggerIsBetter:
                    if testMetric > self.bestMetric:    isBestEpoch = True
                elif testMetric < self.bestMetric:      isBestEpoch = True
                if isBestEpoch:
                    self.bestMetric = testMetric
                    self.bestEpochInfo = epochInfo
                    
            self.updateTrainingTable('AddRow', epochInfo)
            if self.afterEpoch is not None:
                self.afterEpoch(*epochInfo)

            if self.minLoss is not None:
                if loss<=self.minLoss:      # Stop if it is good enough and not improving.
                    numMinLoss += 1
                    if numMinLoss>=self.minLossCount:
                        self.trainTime = time.time() - startTime
                        if self.saveModelFileName is not None:
                            self.updateTrainingTable('  Saving model to "%s" ...'%(self.saveModelFileName))
                            self.save(self.saveModelFileName, epochInfo=epochInfo)
                        self.updateTrainingTable('End', "Stopped. Loss is small enough: %s <= %s"%(str(loss), str(self.minLoss)))
                        return
        
            if self.maxLoss is not None:
                if loss>self.maxLoss:       # Stop if it is too bad.
                    self.trainTime = time.time() - startTime
                    if self.saveModelFileName is not None:
                        self.updateTrainingTable('  Saving model to "%s" ...'%(self.saveModelFileName))
                        self.save(self.saveModelFileName, epochInfo=epochInfo)
                    self.updateTrainingTable('End', "Stopped. Loss too large: %s > %s"%(str(loss), str(self.maxLoss)))
                    return

            if self.saveModelFileName is not None:
                if self.saveBest and isBestEpoch:
                    if self.saveModelFileName[-4:].lower()=='.fbm':
                        bestModelName = self.saveModelFileName[:-4]+'Best'+self.saveModelFileName[-4:]
                    else:
                        bestModelName = (self.saveModelFileName + 'Best')
                    self.updateTrainingTable('  Saving model to "%s" ...'%(bestModelName))
                    self.save(bestModelName, epochInfo=epochInfo)

                if (((epoch+1) % self.savePeriod)==0) or ((epoch+1)==self.numEpochs):
                    self.updateTrainingTable('  Saving model to "%s" ...'%(self.saveModelFileName))
                    self.save(self.saveModelFileName, epochInfo=epochInfo)
            
            epoch += 1

        self.trainTime = time.time() - startTime
        self.updateTrainingTable('End')

    # ******************************************************************************************************************
    def evaluateDSet(self, dataSet, batchSize=None, quiet=False, returnMetric=False, **kwargs):
        r"""
        This function evaluates this model using the specified dataset.
        
        Parameters
        ----------
        dataSet : dataset object (Derived from "BaseDSet" class)
            The dataset object that is used for the evaluation.
            
        batchSize : int
            The batchSize used for the evaluation process. This function processes one batch of the samples at a time. If this is None, the batch size specified by the dataset object is used instead.
            
        quiet : Boolean
            If true, no messages are printed to the "stdout" during the evaluation process.

        returnMetric : Boolean
            If true, instead of calculating all the results, just calculates the main metric of the dataset and returns that. This is mostly used during the training at the end of each epoch.
            
            Otherwise, if this is False (the default), the full results are calculated and a dictionary of all results is returned.
        
        **kwargs : dict
            This contains some additional task specific arguments. All these
            arguments are passed down to the dataset's "evaluateModel" method. Here
            is a list of what can be included in this dictionary.
            
                * **maxSamples (int)**: The max number of samples from the "dataSet" to be processed for the evaluation of the model. If not specified, all samples are used (default behavior).
                    
                * **topK (int)**: For classification cases, this indicates whether a "top-K" accuracy value should also be calculated. For example for ImageNet dataset classification, usually the top-5 accuracy value is used (topK=5) besides the top-1. If it is zero (default), the top-K error is not calculated. This is ignored for regression cases.

                * **confMat (Boolean)**: For classification cases, this indicates whether the confusion matrix should be calculated. If the number of classes is more than 10, this argument is ignored and confusion matrix is not calculated. This is ignored for regression cases.

                * **expAcc (Boolean or None)**: Ignored for regression cases. For classification cases:
                 
                    * If this is a True, the expected accuracy and kappa values are also calculated. When the number of classes and/or number of evaluation samples is large, calculating expected accuracy can take a long time.
                    * If this is False, the expected accuracy and kappa are not calculated.
                    * If this is None (the default), then the expected accuracy and kappa are calculated only if number of classes does not exceed 10.
                                        
                    **Note**: If confMat is True, then expAcc is automatically set to True.

                * **jsonFile (str)**: The name of JSON file that is created by this function. This is used with some NLP applications where the results could be saved to a JSON file for evaluation.

        Returns
        -------
        If returnMetric is True, the actual value of dataset's main metric is returned.
        Otherwise, this function returns a dictionary containing the results of the evaluation process.
        """
        
        if dataSet is None: return None
        if returnMetric:
            return float(dataSet.evaluateModel(self, batchSize, quiet, True, **kwargs))
            
        self.results = dataSet.evaluateModel(self, batchSize, quiet, False, **kwargs)
        return self.results
    
    # ******************************************************************************************************************
    def evaluate(self, quiet=False, **kwargs):
        r"""
        This function evaluates this model using "testDs" dataset that was specified when the model was created.
        
        Parameters
        ----------
        quiet : Boolean
            If true, no messages are printed to the "stdout" during the evaluation process.

        **kwargs : dict
            This contains some additional task specific arguments. All these arguments are passed down to the "evaluateDSet" method. Please refer to the documentation for "evaluateDSet" for a list of possible arguments in "kwargs"
            
        Returns
        -------
        dict
            A dictionary containing the results of the evaluation process.
        """
        assert self.testDs is not None, "The 'evaluate' function can only be used when a test dataset is specified!"
        return self.evaluateDSet(self.testDs, quiet=quiet, **kwargs)
 
    # ******************************************************************************************************************
    def evalBatch(self, batchSamples, batchLabels):
        r"""
        This function is used by the evalMultiDimRegression function defined in the BaseDSet class. It can help improve the evaluation time for some regression problems that support calculating the evaluation metrics as part of tensorflow graph. See the "evalMultiDimRegression" function for more info.
        
        Parameters
        ----------
        batchSamples: numpy array, or tuple of numpy arrays
            The samples used for inference. If number of samples is a multiple of the number of towers, the sub-batches are assigned to each tower and the whole operation is done in one call to "session.run". Otherwise the remaining samples are processed in an additional "session.run" call which uses only the first tower.
        
        batchLabels : numpy array, or tuple of numpy arrays
            The labels used for evaluation of the predicted results. These labels are passed to the output layer supporting evaluation (its "supportsEval" is true), the output layer then calculates the evaluation metrics for this batch of samples.

        Returns
        -------
        Depends on the implementation of the output layer
            The results depends on the implementation of the output layer. Currently the regression output layer "REG" returns a tuple containing sum squared errors and sum absolute errors. These values are passed back to the "evalMultiDimRegression" function which accumulates them for the calculation of other regression evaluation metrics such as MSE, MAE, PSNR.
        """
        assert self.layers.output.supportsEval, "'evalBatch' is not supported by '%s'!"%(self.layers.output.scope)
        
        # Samples can be a numpy array or a tuple of numpy arrays
        remainingSamples = len(batchSamples[0]) if type(batchSamples)==tuple else len(batchSamples)
        samplesPerTower = remainingSamples//len(self.towers)
        results = None
        if samplesPerTower>0:
            feedDic = self.layers.input.feed(batchSamples, self.towers)
            assert feedDic is not None
            feedDic.update( self.layers.output.feed(batchLabels, self.towers) )
            results = self.session.run(self.evalResults, feed_dict=feedDic)
            remainingSamples -= len(self.towers) * samplesPerTower

        if remainingSamples>0:
            rSamples = tuple(x[-remainingSamples:] for x in batchSamples) if type(batchSamples)==tuple else batchSamples[-remainingSamples:]
            feedDic = self.layers.input.feed(rSamples)  # Use last tower for the remaining samples
            assert feedDic is not None
            feedDic.update( self.layers.output.feed(batchLabels) )
            remainingResults = self.session.run(self.towers[-1].evalResults, feed_dict=feedDic)
            if results is None:
                results = remainingResults
            elif type(results) == tuple:
                tupLen = len(results)
                results = tuple( np.concatenate((results[i], remainingResults[i])) for i in range(tupLen) )
            else:
                results = np.concatenate((results, remainingResults))
        
        return results

    # ******************************************************************************************************************
    def inferBatch(self, samples, returnProbs=True):
        r"""
        Runs the model in inference mode using the specified batch of samples and returns the predicted outputs generated by the model. The inference is applied on all the samples in one or two operations. There is no loop going through the samples. The GPUs may run out of memory if all of the samples do not fit into the GPU memory.


        Parameters
        ----------
        samples: numpy array, or tuple of numpy arrays
            The samples used for inference. If number of samples is a multiple of the number of towers, the sub-batches are assigned to each tower and the whole operation is done in one call to "session.run". Otherwise the remaining samples are processed in an additional "session.run" call which uses only the first tower.
            
            If this is a tuple, the input has multiple components. For example for NLP tasks, we have tokenIds and tokenTypes as two numpy arrays packed in a tuple and given to this function for inference.
        
        returnProbs : Boolean
            If true, the predicted probabilities (or Confidence) of each class is returned for each sample. Otherwise, only the predicted classes are returned. This parameter only applies to classification tasks.

        Returns
        -------
        Depends on the type of model and the parameters passed to this function
            The output of the model is returned for each given sample. The output of this function depends on the type of model and the parameters passed to this function. The output layer used in the model determines what type of output is returned as a result of inference.
            
            The output can be a single numpy array containing one sub-tensor for each sample in "samples". It can also be tuple of several numpy arrays each containing different components of the results for each sample. For example for object detection the result could be a tuple containing prediction scores and bounding boxes in different numpy arrays.
        """
        # Samples can be a numpy array or a tuple of numpy arrays
        remainingSamples = len(samples[0]) if type(samples)==tuple else len(samples)
        samplesPerTower = remainingSamples//len(self.towers)
        results = None
        if samplesPerTower>0:
            feedDic = self.layers.input.feed(samples, self.towers)
            if feedDic is not None:
                results = self.session.run(self.tfInferPrediction, feed_dict=feedDic)
                remainingSamples -= len(self.towers) * samplesPerTower

        if remainingSamples>0:
            rSamples = tuple(x[-remainingSamples:] for x in samples) if type(samples)==tuple else samples[-remainingSamples:]
            feedDic = self.layers.input.feed(rSamples)  # Use last tower for the remaining samples
            remainingResults = self.session.run(self.towers[-1].tfInferPrediction, feed_dict=feedDic)
            if results is None:
                results = remainingResults
            elif type(results) == tuple:
                tupLen = len(results)
                results = tuple( np.concatenate((results[i], remainingResults[i])) for i in range(tupLen) )
            else:
                results = np.concatenate((results, remainingResults))
        
        return self.layers.output.postProcessResults(results, returnProbs)

    # ******************************************************************************************************************
    def inferOne(self, sample, returnProbs=False):
        r"""
        Runs inference for a single sample. This packages the single sample as an array of size one and calls the "inferBatch" function.
        
        Parameters
        ----------
        samples: numpy array
            The sample used for inference. Please refer to the documentation of "inferBatch" function for more details.
            
        returnProbs : Boolean
            If true, the predicted probabilities (or Confidence) of each class is returned for the sample. Otherwise, only the predicted class is returned. Ignored for non-classification tasks.
            
        Returns
        -------
        Depends on the type of model and the parameters passed to this function
            The output of the model is returned for the given sample. Please refer to the documentation of "inferBatch" function for more details.
        """
        if type(sample)==tuple:     results = self.inferBatch(tuple(np.array([x]) for x in sample), returnProbs)
        else:                       results = self.inferBatch(np.expand_dims(sample, 0), returnProbs)

        if type(results)==tuple:    return tuple(x[0] for x in results)
        else:                       return results[0]
    
    # ******************************************************************************************************************
    def getGradsForEpoch(self):
        r"""
        This function calculates the gradients of loss with respect to each model parameter for every training sample. The sum of the absolute values of the gradients is then calculated for each model parameter. These values are then normalized so that the maximum value is 1. The returned values are numbers between 0 and 1 giving a measure of importance for each specific network parameter.
        
        This function saves the results of calculations in a file with ".grd" extension. When it is called again, if the the file already exist, there is no need to calculate the gradients info again.
        
        Returns
        -------
        list
            The sum absolute gradients for each network parameter as a list of numpy arrays.
        """
        gradsFileName = None
        if self.modelFilePath is not None:
            gradsFileName = os.path.splitext(self.modelFilePath)[0] + '.grds'
            if os.path.exists(gradsFileName):
                Model.printMsg('  Loading gradients from "%s" ...'%(gradsFileName), False)
                fileDic = np.load(gradsFileName, allow_pickle=True)
                self.sumAbsGrads = [fileDic['grads_L%d'%(i)] for i in range(len(fileDic))]
                Model.printMsg(' Done.')
                assert (len(self.sumAbsGrads)==len(self.towers[0].tfNetParams)), "Invalid grads file! (No of params don't match!)"
                return self.sumAbsGrads

        Model.printMsg('  Calculating network parameter gradients (%d Towers)...'%(len(self.towers)))
        self.sumAbsGrads = [np.zeros(netParam.shape) for netParam in self.towers[0].tfNetParams]
        
        numParams = len(self.towers[0].tfNetParams)
        
        # Gradients need to be calculated for each individual sample. We have "bs" towers; so we can work on "bs"
        # samples at a time. That's why we choose batch size to be the number of towers.
        b = 0                   # Batch Number
        bs = len(self.towers)   # Batch Size
        remainingSamples = self.trainDs.numSamples
        for b, (batchSamples, batchLabels) in enumerate(self.trainDs.batches(bs)):
            feedDic = {}
            for t,tower in enumerate(self.towers):
                feedDic[ tower.tfInputSamples ] = batchSamples[t:t+1]
                feedDic[ tower.tfBatchLabels ] = batchLabels[t:t+1]

            self.updateTrainingTable('  Gradients for sample %d ... '%(b*bs))
            absGrads = self.session.run(self.tfAbsGrads, feed_dict=feedDic)
            for l in range(numParams):
                self.sumAbsGrads[l] += absGrads[l]

        # Normalize the results so that avg is 1 and min is 0
        for l in range(numParams):
            meanGrad = self.sumAbsGrads[l].mean()
            if meanGrad == 0:   self.sumAbsGrads[l] += 1.0    # All 0's -> all 1's
            else:               self.sumAbsGrads[l] /= meanGrad
        
        if gradsFileName is not None:
            Model.printMsg('  Saving gradients to "%s" ...'%(gradsFileName))
            np.savez_compressed(gradsFileName, **{'grads_L%d'%(i): grad for i, grad in enumerate(self.sumAbsGrads)})
            os.rename(gradsFileName+'.npz', gradsFileName)  # The savez_compressed unction always adds '.npz' extension

        return self.sumAbsGrads

    # ******************************************************************************************************************
    def __repr__(self):
        r"""
        A brief text description of the instance.
        
        Returns
        -------
        str
            Returns a text string briefly describing this instance of "Model".
        """
        retStr = '\n%s Instance:'%(self.__class__.__name__)
        retStr += '\n    Name:         ' + self.name
        retStr += '\n    Input Shape:  ' + str(list(self.layers.input.inShape[1:]))
        retStr += '\n    Output Shape: ' + str(self.layers.output.outShape)
        retStr += '\n    Layers:       ' + (str(self.layers.count) if self.layers.count>5 else self.layers.getLayersStr())
        return retStr

    # ******************************************************************************************************************
    def printNetConfig(self):
        r"""
        Prints current configuration of the model including training configuration and training state information.
        """
        print('\nNetwork configuration:')
        print('  Input:                     ' + self.layers.input.getInputStr())
        print('  Output:                    ' + self.layers.output.getOutputStr())
        print('  Network Layers:            %s'%str(self.layers.count))
        print('  Tower Devices:             ' + ', '.join(["CPU%d"%(-gpu-1) if gpu<0 else "GPU%d"%gpu for gpu in self.gpus]))
        print('  Total Network Parameters:  {:,}'.format(self.numParams))
        print('  Total Parameter Tensors:   %d'%(len(self.towers[0].tfNetParams)))
        if self.trainDs is not None:
            numTrainable = len(self.towers[0].allTrainable)
            numNT = len(self.towers[0].nonTransferred)
            print('  Trainable Tensors:         %d'%(numTrainable))
            if numNT>0 and numNT<numTrainable:
                print('  Non-Transfered Tensors:    %d'%(numNT))

        if self.trainDs is not None:        print('  Training Samples:          {:,}'.format(self.trainDs.numSamples))
        if self.testDs is not None:         print('  Test Samples:              {:,}'.format(self.testDs.numSamples))
        if self.validationDs is not None:   print('  Validation Samples:        {:,}'.format(self.validationDs.numSamples))

        if self.trainDs is None:
            print('')
            return

        if self.minLoss is not None and self.minLossCount is not None:
            print('  Max Epochs:                %d (Or if loss<%f, %d times)'%(self.numEpochs,
                                                                               self.minLoss, self.minLossCount))
        else:
            print('  Num Epochs:                %d'%(self.numEpochs))
        if self.batchSize is not None:
            print('  Batch Size:                %d'%(self.batchSize))
        print('  L2 Reg. Factor:            %s'%floatStr(self.regFactor,6))
        print('  Global Drop Rate:          %s'%("Disabled" if self.dropRate == 1 else floatStr(self.dropRate,4)))

        if self.learningRatePieces is not None:
            print('  Learning Rate: (Piecewise)')
            if self.learningRateWarmUp > 0:
                print('    Warm-up Batches:         %d'%(self.learningRateWarmUp))
                print('    Pieces (Batch:value):    %s'%('  '.join("%d:%s"%(b,floatStr(v,13)) for b,v in zip(*self.learningRatePieces))))
            else:
                print('    Pieces (Batch:value):    %s'%('  '.join("%d:%s"%(b,floatStr(v,13)) for b,v in zip(*self.learningRatePieces))))
            if self.trainAllBatch>0 and numNT>0 and numNT<numTrainable:
                print('    Train all parameters     After %d batches'%(self.trainAllBatch))

        elif self.lrDecayStep == 0:
            print('  Learning Rate:             %s'%floatStr(self.learningRateInit,13))
        else:
            print('  Learning Rate: (Exponential Decay)')
            if self.learningRateWarmUp > 0:
                print('    Warm-up Batches:         %d'%(self.learningRateWarmUp))
            print('    Initial Value:           %s'%floatStr(self.learningRateInit,13))
            print('    Final Value:             %s'%floatStr(self.learningRateMin,13))
        print('  Optimizer:                 %s'%(self.optimizer))
        if self.saveModelFileName is not None:
            print('  Save model information to: %s'%(self.saveModelFileName))

        if self.trainingState is not None:
            print('\nTraining State:')
            print('  Last Epoch:                %d'%(self.trainingState['epoch']+1))
            print('  Loss:                      %s'%(floatStr(self.trainingState['loss'],10)))
            if self.trainingState.get('testMetric', None) is not None:
                print('  Test %-22s%s'%(self.trainDs.evalMetricName+':', floatStr(self.trainingState['testMetric'],5)))
            if self.trainingState.get('validMetric', None) is not None:
                print('  Validation %-16s%s'%(self.trainDs.evalMetricName+':', floatStr(self.trainingState['validMetric'],5)))
            if self.trainingState['bestEpoch'] is not None:
                e, b, lr, ls, tm, vm = self.trainingState['bestEpoch']
                if e==self.trainingState['epoch']:
                    print('  Best Epoch:                %d'%(e))
                else:
                    print('  Best Epoch:')
                    print('    Epoch:                   %d'%(e))
                    print('    Batch:                   %d'%(b))
                    print('    Learning Rate:           %s'%(floatStr(lr,10)))
                    print('    Loss:                    %s'%(floatStr(ls,10)))
                    if tm is not None:
                        print('    Test %-20s%s'%(self.trainDs.evalMetricName+':', floatStr(tm,5)))
                    if vm is not None:
                        print('    Validation %-14s%s'%(self.trainDs.evalMetricName+':', floatStr(vm,5)))
        print('')

    # ******************************************************************************************************************
    def printLayersInfo(self):
        r"""
        Prints the structure of network in a table. Each row describes one layer of the network.
        """
        self.layers.printLayersInfo()
        
    # ******************************************************************************************************************
    def getLayerOutputs(self, samples, layer, subLayer=-1):
        r"""
        Feeds the network with the specified sample(s) and for each sample, returns a tensor for outputs of the layer specified by "layer" and "subLayer".
                
        Parameters
        ----------
        samples: numpy array, or tuple of numpy arrays
            The samples that are fed to the network. If only one sample is provided by itself (not in a list), only one tensor will be returned.
            
        layer: int or str
            Specifies the layer whose output tensor will be calculated and returned.
            
            * If this is int, it is the index of the layer starting at 0 for the first layer of the network.
            * If this is string, it must be the name (Scope) of the layer as it is printed by the "printLayersInfo" function.
                  
        subLayer : int
            For multi-Stage layers, "subLayer" specifies stage whose output should be returned. The assignment of sublayers is different depending on types of layers. Here is an example from MobileNetV2::
            
                layer: 'BN:ReLU:CLP_H6:GAP'
                                
            ========   =================================================
            Sublayer   Value
            ========   =================================================
            0          The output of Batch Norm layer
            1          The output of ReLU
            2          The output after CLP (Clip the output to max=6.0)
            3          The output of Global Average Pool.
            -1         Last output (Same as 3 in this case)
            ========   =================================================
            
            Please refer to the "layers.py" file for the details about sublayer
            values for each type of layer.

        Returns
        -------
        numpy array
            The output of the layer specified by "layer" and "subLayer" as tensor for each sample in "samples".
        """
        
        # Note: All layer objects have the graph info related to the last tower only. So, this runs on
        # the last tower. There is no Parallelization. If the number of samples is large the GPU can run
        # out of memory.
        feedDic = self.layers.input.feed(samples)
        return self.layers[layer].inferOut( self.session, feedDic, subLayer )
        
    # ******************************************************************************************************************
    def getLayerByName(self, layerScope):
        r"""
        Finds and returns the layer object specified by the "layerScope".
                
        Parameters
        ----------
        layerScope : str
            Specifies the layer. It must be the name (Scope) of the layer as it is printed by the "printLayersInfo" function.
            
        Returns
        -------
        Layer
            Returns the layer object specified by the "layerScope".
        """
        return self.layers[layer]
    
    # ******************************************************************************************************************
    def getAllNpNetParams(self):
        r"""
        Returns all model parameters as a list of NetParam objects.
        
        Returns
        -------
        list
            A list of NetParam objects (All 'NP' mode)
        """
        return NetParam.toNp(self.towers[0].tfNetParams, self.session)
        
    # ******************************************************************************************************************
    def getAllNetParamValues(self):
        r"""
        Returns all model parameters as a list of numpy arrays.
        
        Returns
        -------
        list
            A list of numpy arrays.
        """
        return NetParam.toNpValues(self.towers[0].tfNetParams, self.session)
        
    # ******************************************************************************************************************
    def getLayerParams(self, layer=None, orgNamesFile=None):
        r"""
        Returns the network parameters for the specified layer.
        
        Parameters
        ----------
        layer: int or str
            Specifies the layer whose parameter are returned.
            
            * If this is int, it is the index of the layer starting at 0 for the first layer of the network.
            * If this is string, it must be the name (Scope) of the layer as it is printed by the "printLayersInfo" function.
            * If this is None, then all network parameters are returned in a list of pairs (name, param).

        orgNamesFile : str
            If specified, it must contain the path to the yaml file that was used to import the model from the original h5 file. In this case the names used in the returned parameter info are extracted from the specified file. This is only used if "layer" is set to None.

        Returns
        -------
        list
            A list of numpy tensors for the network parameters of the specified layer. If a layer is not specified, it returns a list of tuples of the form (name, param) for all the network parameters.
        """
        
        if layer is None:
            if orgNamesFile is None:    # Use the names of parameters from the model
                # Return all parameters in a list of tuples
                layerParams = self.getAllNetParamValues()
                
                netParams = []
                for layer in self.layers:
                    layerParamNames = layer.getAllParamStrs()
                    n = len(layerParamNames)
                    netParams += zip(layerParamNames, layerParams[:n])
                    layerParams = layerParams[n:]
                return netParams
        
            # Use the names of parameters from the specified file
            import yaml
            with open(orgNamesFile, 'r') as namesFile:
                orderedNames = yaml.load(namesFile, Loader=yaml.FullLoader)
            orgNamesInfo = []
            for layerName, layerInfo in orderedNames:
                for weightName in layerInfo:
                    weightInfo = None
                    if type(weightName)==list: weightName, weightInfo = weightName
                    if type(weightInfo)==list:
                        for subWeightName in weightInfo:
                            if type(subWeightName)==list: subWeightName = subWeightName[0]
                            orgNamesInfo += [(layerName, '%s/%s' % (weightName, subWeightName))]
                    else:
                        orgNamesInfo += [(layerName, weightName)]

            layerParams = self.getAllNetParamValues()
            netParams = []
            op = 0      # Original Param Index
            for layer in self.layers:
                if layer.isDecomposed():
                    orgLayerName, orgWeightName = orgNamesInfo[op]
                    layerParamNames = [ '%s/%s_G'%(orgLayerName, orgWeightName),
                                        '%s/%s_H'%(orgLayerName, orgWeightName) ]
                    if layer.hasBias:
                        orgLayerName, orgBiasName = orgNamesInfo[op+1]
                        layerParamNames += [ '%s/%s'%(orgLayerName, orgBiasName) ]
                    op += 2
                    n = len(layerParamNames)
                else:
                    n = len(layer.tfParams)
                    layerParamNames = []
                    for i in range(n):  layerParamNames += [ '%s/%s'%(orgNamesInfo[op+i][0], orgNamesInfo[op+i][1]) ]
                    op += n
                netParams += zip(layerParamNames, layerParams[:n])
                layerParams = layerParams[n:]
                print( layerParamNames )
            return netParams

        return self.layers[layer].netParamValues

    # ******************************************************************************************************************
    def setLayerParams(self, layer, params):
        r"""
        Modifies the network parameters for the specified layer.
        
        Parameters
        ----------
        layer: int or str
            Specifies the layer whose parameter are returned.
            
            * If this is int, it is the index of the layer starting at 0 for the first layer of the network.
            * If this is string, it must be the name (Scope) of the layer as it is printed by the "printLayersInfo" function.

        params: list of numpy arrays
            A list of numpy arrays for each parameter of the specified layer. The length of this list must match the actual number of parameters in the specified layer.
        """

        layer = self.layers[layer]

        paramIndex = layer.paramIndex
        numParams = len(layer.netParams)
        assert numParams == len(params), "Layer \"%s\" requires %d tensor, %d given!"%(layer, numParams, len(params))
        for tower in self.towers:
            for i in range(numParams):
                tower.tfNetParams[paramIndex+i].param.load(params[i], self.session)

    # ******************************************************************************************************************
    def freezeLayerParams(self, layers):
        r"""
        Makes the parameters of specified layers non-trainable. This is not reversible.
        
        Parameters
        ----------
        layers: tuple or list of strings
            Specifies the layers to freeze:
            
            * If this is a tuple, it must have 2 strings specifying the first and last layer to freeze.
            * If this is a list, it must contain the strings specifying the layers to freeze.
        """
        numFreezed = None
        for tower in self.towers:
            if numFreezed is None:
                numFreezed = tower.freezeLayerParams(layers)
            else:
                assert numFreezed == tower.freezeLayerParams(layers), "Different towers froze different parameters!"
        Model.printMsg("Total number of parameters frozen: %d"%(numFreezed))
            
    # ******************************************************************************************************************
    def closeSession(self):
        r"""
        Closes current TensorFlow Session. Call this after you are done with current instance of this class.
        """
        if self.session is not None:
            self.session.close()

    # ******************************************************************************************************************
    def setCallback(self, afterBatch=None, afterEpoch=None):
        r"""
        Sets the callback functions that are called at the end of each epoch or batch.

        Parameters
        ----------
        afterBatch : function
            This function is called after each batch of training data is processed. The following parameters are passed to this function:
            
            * **epoch**: Current epoch number. (starting from 0)
            * **batch**: Batch number in current epoch. (starting from 0)
            * **learningRate**: The value of learning rate for this batch.
            * **loss**: The loss value for this batch.
                
        afterEpoch : function
            This function is called at the end of each training epoch. The
            following parameters are passed to this function:
            
            * **epoch**: Current epoch number.
            * **batch**: Batch number of the last batch in current epoch.
            * **learningRate**: The value of learning rate for the last batch of current epoch.
            * **loss**: The loss value for the last batch of current epoch.
            * **testMetric**: The evaluation metric value (ErrorRate/Accuracy/MSE/mAP calculated on test dataset after the end of this epoch.
            * **validMetric**: The evaluation metric value (ErrorRate/Accuracy/MSE/mAP calculated on validation dataset after the end of this epoch.
        """
        self.afterBatch = afterBatch
        self.afterEpoch = afterEpoch
    
    # ******************************************************************************************************************
    @classmethod
    def printMsg(cls, textStr, eol=True):
        r"""
        print the specified text string only if the quiet flag is not set.
        
        Parameters
        ----------
        textStr : str
            A text string.
            
        eol: Boolean
            True means append the text with an end of line character.
        """
        if cls.quiet: return
        if eol == False:    sys.stdout.write(textStr)
        else:               sys.stdout.write(textStr+'\n')
        sys.stdout.flush()
    
    # ******************************************************************************************************************
    def updateTrainingTable(self, code, data=None):
        r"""
        Updates the training table by printing new rows during the training. If the "quiet" flag is set, this function returns immediately without printing anything.

        Parameters
        ----------
        code : str
            Specifies the type of update to the table. One of the following:
            
            * "Start":        Start of the table. Prints the table header.
            * "Separator":    Draws a horizontal separator line.
            * "Batch":        Called at the end of each batch. Prints current batch information.
            * "AddRow":       Called at the end of each epoch. Adds one row to the table containing the epoch information.
            * "End":          Called at the end of training. Closes the table.
            * Anything else:  If "code" is not any of the above, the function just prints the given text.

        data : Different types, default: None
            The value of "data" depends on the code:
            
            * "Start":        Ignored.
            * "Separator":    Ignored.
            * "Batch":        A tuple (epoch, batch) containing the epoch number and batch number.
            * "AddRow":       A tuple (epoch, batch, learningRate, loss, validMetric, testMetric) containing the epoch number, batch number, learning rate, loss, validation metric, and test metric for the epoch.
            * "End":          Optional text value. The text will be printed on the line immediately following the table.
            * Anything else:  If not None, a new-line character is also print.
        """
        if self.quiet:  return

        if code.lower() == 'start':
            self.tableHeaderCounter = -1
            self.tableMaxBatch = 0
            return

        #        123456   1234567   1234567890123   123456789   12345678901234567
        sep = '+--------+---------+---------------+-----------+-------------------+'
        if code.lower()=='separator':
            print(sep)
            return

        if code.lower()=='batch':
            epoch, batch = data
            if batch>self.tableMaxBatch:
                self.tableMaxBatch=batch
                self.printMsg('  Epoch %d, Batch %d     \r'%(epoch, batch), False)
            else:
                batchStr = '  Epoch %d, Batch %d/%d '%(epoch, batch, self.tableMaxBatch)
                remainingLen = len(sep)-len(batchStr)
                numProgressChars = batch*remainingLen//self.tableMaxBatch
                batchStr += '█'*numProgressChars
                batchStr += '.'*(len(sep)-len(batchStr)) + '\r'
                Model.printMsg(batchStr, False)
            return

        if code.lower()=='addrow':
            self.tableHeaderCounter = (self.tableHeaderCounter+1)%50
            metricStr = 'Acc.' if (self.trainDs.evalMetricName == 'Accuracy') else self.trainDs.evalMetricName
            if self.tableHeaderCounter==0:
                #            123456   1234567   1234567890123   123456789   12345678901234567
                print('%s\n| Epoch  | Batch   | Learning Rate | Loss      | Valid/Test %-6s |\n%s'%(sep,metricStr,sep))
            
            epoch, batch, learningRate, loss, validMetric, testMetric = data
            rowStr = '| %-6s | %-7s | %s | %s |' % (str(epoch+1),
                                                    str(abs(batch)),
                                                    floatStr(learningRate,13),
                                                    floatStr(loss,9))
            isPercent = metricStr in ['mAP', 'Acc.', 'Error']
            #                                        12345678
            if validMetric is None:     rowStr +=  ' N/A     '
            elif isPercent:             rowStr +=  ' %7.2f%%'%(validMetric)    # Include a percent sign
            else:                       rowStr +=  ' %-8.3f'%(validMetric)

            #                                       901234567
            if testMetric is None:      rowStr +=  ' N/A      |'
            elif isPercent:             rowStr +=  ' %7.2f%% |'%(testMetric)   # Include a percent sign
            else:                       rowStr +=  ' %-8.3f |'%(testMetric)

            print(rowStr)
            return
            
        if code.lower()=='end':
            print(sep)
            if data is not None:    print(data)
            print('Total Training Time: %.2f Seconds'%(self.trainTime))
            return

        # Otherwise print the message in the "data"
        eol = (data is not None)    # Default: no new line)
        msgStr = code + " "*(len(sep)-len(code)) + ("" if eol else "\r")
        Model.printMsg(msgStr, eol)

    # ******************************************************************************************************************
    @classmethod
    def registerLayerClass(cls, layerClass):
        r"""
        Register a new layer class with fireball. This must be called before a Model is instantiated.
        
        Parameters
        ----------
        layerClass : class
            A class derived from the "Layer" class.
        """
        assert layerClass.name.lower() not in Layer.layerClasses, \
               "The layer '%s' has already been registered!"%(layerClass.name)
        Layer.numCustomLayers += 1
        Layer.layerClasses[ layerClass.name.lower() ] = (layerClass, 1000 + Layer.numCustomLayers)
