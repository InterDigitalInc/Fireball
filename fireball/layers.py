# Copyright (c) 2019-2020 InterDigital AI Research Lab
"""
This file contains the implementation for all fireball layers. For more information
about the layers supported in Fireball, please refer to the documentation for the
function "Layers::__init__" below.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 03/04/2019    Shahab Hamidi-Rad       Created the file.
# 08/25/2020    Shahab                  Started documenting the revision history for this file.
#                                       Support for input/output layers and BERT was added in fireball version 1.3.
# 10/29/2020    Shahab                  Added support for export to TensorFlow (see the getTfText functions).
# 11/29/2020    Shahab                  Reorganized the code for the TensorFlow export (See the buildTf functions).
#                                       Also added support for ONNX export (See the buildOnnx functions).
#                                       Support for export to CoreML was also updated. Now the buildCml functions
#                                       handle the CoreML export functionality for each type of layer.
# **********************************************************************************************************************
import numpy as np

import tensorflow as tf
try:    import tensorflow.compat.v1 as tf1
except: tf1 = tf
try:    import tensorflow.random as tfr
except: tfr = tf

from . import ldr

from .fb2tf import *
from .netparam import NetParam

# **********************************************************************************************************************
# Using this forces the same random numbers across different tensorflow sessions
SEED = 36478                    # Set to None for random seed.
DEFAULT_ACTIVATION = 'none'
DO_CHECK_NUMERICS = True

# **********************************************************************************************************************
def checkNumeric(tensors, text):
    if not DO_CHECK_NUMERICS:   return tensors
    if type(tensors) != tuple:
        if tensors.dtype not in [tf.float32, tf.float64]:   return tensors
        return tf.debugging.check_numerics(tensors, text)
    
    return tuple( checkNumeric(tensor, text) for tensor in tensors )
    
# **********************************************************************************************************************
def applyPadding(inShape, kernel, stride, padding, dilation=[1,1]):
    kernelX, kernelY = kernel
    strideX, strideY = stride
    dilationX, dilationY = dilation

    def getOutDim(inDim, kernel, stride, dilation=1, pads=[0,0]):
        return int(np.ceil( (inDim + pads[0] + pads[1] - (kernel-1)*dilation)/float(stride) ))

    if padding == 'valid':
        return [ getOutDim(inShape[0], kernelY, strideY, dilationY),
                 getOutDim(inShape[1], kernelX, strideX, dilationX),
                 inShape[2] ]
    
    if padding == 'same':
        return [ inShape[0]//strideY, inShape[1]//strideX, inShape[2] ]
    
    # Note: padding is [[top, bottom], [left,right]]
    return [ getOutDim(inShape[0], kernelY, strideY, dilationY, padding[0]),
             getOutDim(inShape[1], kernelX, strideX, dilationX, padding[1]),
             inShape[2] ]

# **********************************************************************************************************************
def gelu(x, name):
    # Gaussian Error Linear Unit. (A smoother version of the RELU: https://arxiv.org/abs/1606.08415)
    # Original implementation:
    #   https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L264
    with tf.name_scope(name):
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        output = x * cdf
    return output

# **********************************************************************************************************************
class Layers:
    def __init__(self, layersStr, blocks, model):
        """
        __init__
        This function receives a text string containing layers information and a list of blocks and creates an
        array of "Layer" objects.
        
        Arguments:
            layersStr: Required, string
                The layer(s) information in text string. It contains the information for all the layers.
            blocks: Required, list
                The list of Block object. The blocks defined in the list can be used as a layer in the layersStr.

        Returns:
            A list of layer objects containing information for each layer of the network.
        
        The Layer Info String has the following format:
            "stage1;stage2;...;stageN"
            
            * If there are no semicolons, there is only one stage containing all layers.
            * Stages can be used to group layers.
            * Each Stage has the following format:
                layer1,layer2,...layerM
                
                * Each layer has the following format:
                    [i>]layerInfo:activation:postActivation1:postActivation2:...:postActivationP[>o]
                    * i>, >o (Input/output Netmarks]:
                        The netmarks are used to connect layers to each other in a non-sequential way. Each
                        netmark is specified by a unique number.
                        If you want the output a layer to be used later as input to a different layer, you
                        add ">o" to the end of layer specification where o is a unique integer specifying
                        the netmark.
                        When you want a layer to have an input other than previous layers output, you can
                        include an input netmark by prefixing the layer with "i>" where i specifies the netmark
                        specifying the output of other network layer.
                        Netmarks can also be used in some postActivations. See "ADD, SEL, etc." for example.
                        
                    * layerInfo has the following format:
                        layerName_layerParam1_layerParam2_...layerParamK
                        
                        * layerName can be one of:
                            Input Layers:
                                IMG: Image Input
                                TENSOR: Tensor Input
                                EMB: Embedding Input (NLP tasks)
                            Hidden Layers:
                                FC: Fully Connected
                                CONV: Convolutional
                                DWCN: Depth-wise convolution
                                BN: Batch Normalization
                                LN: Layer Normalization
                                AFM: Aggregate Feature Maps (Object Detection)
                                BERT: Bidirectional Encoder Representations from Transformers (NLP)
                            Output Layers:
                                CLASS: Classification Output
                                REG: Regression Output
                                OBJECT: Object Detection Output
                                ANSWER: Answer Output (NLP Question/Answering tasks)
                            <BlockName>: A predefined block name
                        * The first layer of the network MUST be an Input layer.
                        * The last layer of the network MUST be an Output layer.
                        * layerName is case insensitive
                        * Each layerParam has the following format:
                            <paramLetter><paramValue>
                            * paramLetter is a single letter (A-Z case insensitive) specifying the parameter.
                            * paramValue an alpha numeric string giving the value of parameter.
                        * layerParams can come in any order.
                        * Examples for layerInfo:
                            a) FC_O128          ->  A Fully Connected layer with output size 128.
                            b) CONV_K3_S1_Ps    ->  A Convolutional layer with kernel size 3, stride 1, and padding "Same"
                    * "activation" can be one of the following activation functions:
                        ReLU:   Rectified Linear Unit
                        GeLU:   Gaussian Error Linear Unit
                        Tanh:   Tangent Hyperbolic
                        Sig:    Sigmoid
                        Soft:   Softmax
                        None:   No Activation
                    * "activation" is case insensitive
                    * If "activation" is missing the default is "None".
                    * "activation" must be specified if there are any "postActivation" (See below)
                    * "postActivation" has the following format:
                        paName_paParam1_paParam2_...paParamL
                        * paName can be one of:
                            MP: Max Pooling
                            AP: Average Pooling
                            TP: Transformer pooling (NLP)
                            GAP: Global Average Pooling
                            UP: Upsampling
                            DO: Dropout
                            CLP: Clip
                            L2R: L2 Regularization
                            FM: Feature Map (Object Detection)
                            ADD: Add the output of current layer with the specified netmarks and output the sum
                            SEL: Select and output of one of the specified netmarks based on the output of current layer
                            WSUM: Output sum of specified netmarks weighted by the output of current layer
                            TUP: Create and output a tuple containing output of current layer and the specified netmarks
                        * paParams have the same format as layerParams defined above.
                        * paParams can come in any order.
                    * P can be zero. (which means a layer with no postActivation)

        * Layer Parameters:
            * FC (Fully Connected)
                - O: outSize, integer, required (Size of the output)
                - B: hasBias, integer, default=1 (0 means no biases for this layer)
                
                Decomposed layer params (Ignored if not decomposed)
                  - R: rank, integer, default=0, Non-zero for decomposed layers
                  - L: decomposition type, str, default='lr', can be 'lr' or 'ldr' for decomposed layers.
                  - E: The 'e' value for LDR layers, default=-2, ignored if not an LDR decomposed layer
                  - F: The 'f' value for LDR layers, default=2, ignored if not an LDR decomposed layer
            * CONV (Convolutional)
                - K: kernel, integer/Pair of integers, required (kernel size)
                - S: stride, integer/Pair of integers, default=1 (stride)
                - P: padding, 's', 'v', or numbers in the form of P, XxY, or LEFTxRIGHTxTOPxBOTTOM default=v
                    Examples:
                        P2 =>       pad 2 zeros to all 4 sides of input map
                        P2x3 =>     pad 2 zeros to left and right, 3 zeros to top and bottom
                        P1x2x3x4 => pad 1 zero to left, 2 to right, 3 to top, and 4 to bottom
                - O: outDept, integer, required (number of output channels)
                - B: hasBias, integer, default=1 (0 means no biases for this layer)
                - D: dilation, integer/Pair of integers, default=1
                Decomposed layer params (Ignored if not decomposed)
                  - R: rank, integer, default=0, Non-zero for decomposed layers
                  - L: decomposition type, str, default='lr', can be 'lr' or 'ldr' for decomposed layers.
                  - E: The 'e' value for LDR layers, default=-2, ignored if not an LDR decomposed layer
                  - F: The 'f' value for LDR layers, default=2, ignored if not an LDR decomposed layer
            * DWCN (Depth-wise Convolutional Layer)
                - K: kernel, integer/Pair of integers, required (kernel size)
                - S: stride, integer/Pair of integers, default=1 (stride)
                - P: padding, 's', 'v', or numbers in the form of P, XxY, or LEFTxRIGHTxTOPxBOTTOM default=v (padding type)
                    Examples:
                        P2 =>       pad 2 zeros to all 4 sides of input map
                        P2x3 =>     pad 2 zeros to left and right, 3 zeros to top and bottom
                        P1x2x3x4 => pad 1 zero to left, 2 to right, 3 to top, and 4 to bottom
                - B: hasBias, integer, default=1 (0 means no biases for this layer)
                Decomposed layer params (Ignored if not decomposed)
                  - R: rank, integer, default=0, Non-zero for decomposed layers
                  - L: decomposition type, str, default='lr', can be 'lr' or 'ldr' for decomposed layers.
                  - E: The 'e' value for LDR layers, default=-2, ignored if not an LDR decomposed layer
                  - F: The 'f' value for LDR layers, default=2, ignored if not an LDR decomposed layer
            * BN (Batch Normalization)
                - E: epsilon, float, default=.001
            * LN (Layer Normalization)
                - E: epsilon, float, default=1.0e-12
            * AFM (Aggregate Feature Maps)
                - C: numClasses, integer, required (Number of classes including the background class)
                - T: type, str, default='ssd', (Only SSD type is currently supported)
            * BERT (BERT: Bidirectional Encoder Representations from Transformers)
                - O: outSize, size of Hidden States, required
                - I: intermediateSize, size of the intermediate layer output, required
                - H: numHeads, Number of attention heads, default=12
                - D: dropRate, Dropout Probability, default=0.1
                - S: initStd, Standard Dev. for initializers, default=0.02
                - E: epsilon', The epsilon value for for layer normalizations, default=1e-12
            * ID (Identity)
                ID layer has no parameters
                This is usually used in the block definitions to implement the shortcut path.
                
            * IMG (Image Input)
                - S: image size, integer/Pair of integers (Pair of integer specify widthxheight) required.
                     Examples:
                        S224 =>     Inputs are 224x224 images.
                        S640x480 => Inputs are 640x480 images (width=640, height=480). Note that the "shape" of
                                    input in this case is (480, 640)
                - D: image depth, default=3 (color image), use 1 for monochrome image.
            * TENSOR (Tensor Input)
                - S: shape, list of integers, required, shape of input samples not including the batch dimension
                     Examples:
                        S10 => vector of length 10
                        S3/5 => Matrix with shape= (3,5)
                        S2/4/7 => Tensor with shape=(2,4,7)
            * EMB (Embedding Input)
                - T: Type, str, default='bert', Currently 'bert' and 'sig' embeddings are supported.
                - O: output size of the embedded data
                - S: initStd, Standard Dev. for initializers, default=0.02
                - L: maxLen, integer, default=512 for 'bert' anf 4096 for 'sig', max sequence length
                - V: vocabSize, integer, default=30522 for 'bert' and 64 for 'sig', size of vocabulary
                - R: rank, integer, default=0, Non-zero for decomposed layers, currently only word embedding
                     tensor can be decomposed and only low rank decomposition is supported
            * CLASS (Classification output)
                - C: numClasses, integer, required
            * REG (Regression output)
                - S: output shape, list of integers, required, shape of the output tensors (not including the batch
                     dimension)
                     Examples:
                        S32/32/3 => tensor of shape=[32,32,3] (Can be a 32x32 RGB image for example)
                        S1 => A single value regression, output is a number
            * ANSWER (Answer output for NLP question answering)
                This output layer has no parameters.
            * OBJECT (Object Detection output)
                This output layer has no parameters. This usually follows an AFM layer.

            * In addition to the above hardcoded layers, you can also use any custom defined layers using Blocks. Each
              Block is defined giving the name, parameter definitions, and path definitions. See the Block class for
              more information.
              
        * Post Activation Parameters:
            * MP (Max Pooling)
                - K: kernel, integer/Pair of integers, required (kernel size)
                - S: stride, integer/Pair of integers, default=kernel (stride)
                - P: padding, 's', 'v', or numbers in the form of P, XxY, or LEFTxRIGHTxTOPxBOTTOM default=v (padding type)
                    Examples:
                        P2 =>       pad 2 zeros to all 4 sides of input map
                        P2x3 =>     pad 2 zeros to left and right, 3 zeros to top and bottom
                        P1x2x3x4 => pad 1 zero to left, 2 to right, 3 to top, and 4 to bottom
            * AP (Average Pooling)
                - K: kernel, integer/Pair of integers, required (kernel size)
                - S: stride, integer/Pair of integers, default=kernel (stride)
                - P: padding, string or integer/Pair of integers, default=v (padding type)
                    It can be 's' for "same" padding, 'v' for "valid" padding, or a number/pairs of numbers for padding
                    along x and y axis.
            * GAP (Global Average Pooling)
                GAP has no parameters
            * TP: (Transformer Pooling)
                - N: numVectors, integer, default=1
            * UP (Upsampling)
                - S: scale, integer/Pair of integers, required (scale of upsampling)
            * DO (Dropout)
                - R: dropRate, float, required (probability (or rate) of dropout)
            * CLP (Clip by value or L2 norm)
                - H: Higher Bound, float, default=+inf
                - L: Lower Bound, float, default=-inf
                - N: Norm Value, float, default=+inf
                Note: At least one of H, L, or N must be specified.
            * L2R (L2 Regularization)
                - F: factor, float, default=1.0 (The L2 regularization factor)
                    If the factor is not specified, it uses 1. In this case the Model's regularization factor must be
                    specified.
                    The final regularization term added to the loss value is calculated as follows:
                        reg = Model.regFactor * (f1*reg1 + f2*reg2 + ...)
                Note that technically this is not a post activation process because the results is applied to the loss
                rather than the specified layer.
            * FM (Feature Map)
                - A: Anchors, integer, required,  The number of anchor boxes for this feature map
                - N: Normalization, integer, default=0, Currently only N2 is acceptable. default is no normalization.
            * ADD:
                - N: A list of integers specifying the netmarks (separated by '/'). Example: ADD_N2/3
            * SEL:
                - N: A list of integers specifying the netmarks (separated by '/'). Example: ADD_N2/3/4
            * WSUM:
                - N: A list of integers specifying the netmarks (separated by '/'). Example: ADD_N2/3
            * TUP:
                - N: A list of integers specifying the netmarks (separated by '/'). Example: ADD_N2
        
        Note: When a pair of numbers is specified for a parameter (Kernel, Stride, Padding, Size, etc.), the numbers
              are separated by 'x', the first number is the horizontal component, and the second number is the vertical
              component. For example "K1x3" can be used for 1D convolutions with kernel size 3 which can be applied to
              vertical images (i.e. vectors) of size 1xn (or shape (n,1)). Note that this is different from "shape"
              of kernel which in this case is (3,1).
        """
        self.paFms = []
        self.netmarks = {}
        self.blocks = blocks
        self.model = model
        self.attentionMasks = None
        
        refBlocks = { block.name.lower(): block for block in blocks }
        stageStrs = layersStr.split(';')
        numStages = len(stageStrs)
        outLayer = None
        layerIndex = 0
        self.layers = []
        stage = -1
        for stageStr in stageStrs:
            stage += 1
            layerStrs = stageStr.split(',')
            localLayerIndex = 1
            for layerStr in layerStrs:
                numSubLayers = 1
                if '*' in layerStr:     numSubLayers, layerStr = layerStr.split('*')
                for i in range(int(numSubLayers)):
                    if (layerIndex==1) and (stage==0): stage = 1    # Make sure first non-input layer has stage 1
                    scope = 'S%d_L%d'%(stage, localLayerIndex) if numStages>1 else 'L%d'%(localLayerIndex)
                    layer = self.getLayer(layerIndex, scope, layerStr, refBlocks)
                    
                    if layer.isInput:       layer.scope = "IN_" + layer.name
                    elif layer.isOutput:    layer.scope = "OUT_" + layer.name
                    else:                   localLayerIndex += 1

                    if numStages>1: layer.stage = stage
                    layerIndex += 1
                    self.layers += [layer]

    # ******************************************************************************************************************
    def getLayer(self, index, scope, layerStr, refBlocks={}, parent=None):
        layerStr = layerStr.strip(' ')
        
        outNum = inNum = None
        if '>' in layerStr:
            parts = layerStr.split('>')
            if len(parts)!=2:   raise ValueError('Syntax Error in "%s"!'%(layerStr))
            try:
                inNum = int(parts[0])
                layerStr = parts[1]
            except:
                outNum = int(parts[1])
                layerStr = parts[0]

        subLayers = layerStr.lower().split(':')

        layerInfo = subLayers[0].split('_')
        layerType = layerInfo[0]
        layerArgs = layerInfo[1:] if len(layerInfo)>1 else []
        
        actStr = subLayers[1] if len(subLayers)>1 else DEFAULT_ACTIVATION
        if actStr.strip()=="":  actStr = DEFAULT_ACTIVATION
        postActivationStrs = subLayers[2:] if len(subLayers)>2 else []
        
        # Note the only case that we return None for a layer if for the case of ID. (for example the shortcut in resnet)
        layer = None
        if layerType in Layer.layerClasses:
            layerClass = Layer.layerClasses[layerType][0]
            if layerClass is not None:
                layer = layerClass(self, index, scope, layerArgs, actStr, parent)
                layer.addPostActivations(postActivationStrs)
            else:
                assert layerType == "id", "No class defined for layer '%s'!"%(layerType)
                return None

        elif layerType in refBlocks:
            layer = refBlocks[layerType].instantiate(self, index, scope, layerArgs, actStr, parent)
            layer.addPostActivations(postActivationStrs)

        else:
            raise ValueError('Unknown layer "%s"!'%(layerType.upper()))

        
        if outNum is not None:
            if outNum in self.netmarks: raise ValueError('Netmark %d has already been used!'%(outNum))
            self.netmarks[ outNum ] = layer
        layer.outNetmark = outNum
        layer.inNetmark = inNum
        
        return layer

    # ******************************************************************************************************************
    def __getitem__(self, key):
        if type(key) == int:  return self.layers[key]

        for layer in self.layers:
            if layer.scope == key:
                return layer
        return None
        
    # ******************************************************************************************************************
    def __iter__(self):
        for layer in self.layers:
            yield layer

    # ******************************************************************************************************************
    @property
    def count(self):
        return len(self.layers)

    # ******************************************************************************************************************
    @property
    def input(self):
        return self.layers[0]
        
    # ******************************************************************************************************************
    @property
    def output(self):
        return self.layers[-1]

    # ******************************************************************************************************************
    def getLayersStr(self):
        layersStr = ''
        prevStage = 0
        for layer in self.layers:
            layerStr = layer.getLayerStr()
            
            if layer.inNetmark is not None:     layerStr = "%d>"%(layer.inNetmark) + layerStr
            if layer.outNetmark is not None:    layerStr += ">%d"%(layer.outNetmark)

            if layer.stage != prevStage:    layersStr += ';' + layerStr
            else:                           layersStr += ',' + layerStr
            prevStage = layer.stage
        return layersStr[1:]

    # ******************************************************************************************************************
    def setAllShapes(self):
        outShape = None
        for layer in self.layers:
            inShape = outShape if layer.inNetmark is None else self.netmarks[ layer.inNetmark ].outShape
            outShape = layer.getOutShape(inShape)

    # ******************************************************************************************************************
    def getNameNetParams(self):
        """
        Returns a list of tuples (name, netParam) for all parameters of the network.
        """
        nameNetParams = []
        for layer in self.layers:
            layerParamNames = layer.getAllParamStrs(includeSizeInfo=False)
            layerNetParams = NetParam.toNp(layer.netParams, self.model.session)
            for i, name in enumerate(layerParamNames):
                nameNetParams += [ (name, layerNetParams[i]) ]
        return nameNetParams
        
    # ******************************************************************************************************************
    def printAllParamInfo(self):
        """
        printAllParamInfo
        Prints parameter info for all layers. Prints the parameters in layers inside blocks.
        """
        print('')
        for layer in self.layers:
            lineStrs = layer.getAllParamStrs(includeSizeInfo=True)
            for lineStr in lineStrs:    print(lineStr)

    # ******************************************************************************************************************
    def printLayersInfo(self):
        """
        printLayersInfo
        Prints the structure of network in a table. Each row describes one layer of the network.
        """
        print('')
        #      123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345
        print('Scope            InShape       Comments                 OutShape      Activ.   Post Act.        # of Params')
        print('---------------  ------------  -----------------------  ------------  -------  ---------------  -----------')
        #     '123456789012345  123456789012  12345678901234567890123  123456789012  1234567  123456789012345  12345678901'
        
        def getShapeStr(shp):
            if shp is None:         return ''
            if len(shp)==1:         return str(shp[0])
            elif len(shp)==2:
                # This is for layers in BERT models. It is (seqLen, outSize) except for the ANSWER
                # output which is (2, seqLen). When seqLen is variable we use "≤X" where X is max sequence length.
                if shp[0]==-1:      return '≤%d %d'%(self.input.maxLen, shp[1]) # The case for variable seqLen
                elif shp[1]==-1:    return '%d ≤%d'%(shp[0],self.input.maxLen)  # The case for ANSWER output
                else:               return '%d %d'%(shp[0], shp[1])
            elif len(shp)==3:       return '%d %d %d'%(shp[0], shp[1], shp[2])
            else:                   raise ValueError('Invalid Shape %s!'%(str(shp)))

        totalParams = 0
        for layer in self.layers:
            lineStr = ""
            if layer.inNetmark is not None: lineStr += '%d>\n'%(layer.inNetmark)
            if layer.outNetmark is not None:
                outPart = ">%d"%(layer.outNetmark)
                lineStr += ('%%-%ds%%s  '%(15-len(outPart)))%(layer.scope, outPart)
            else:
                lineStr += '%-15s  '%(layer.scope)
            lineStr += '%-12s  '%(getShapeStr(layer.inShape))
            lineStr += '%-23s  '%(layer.getShortDesc())
            lineStr += '%-12s  '%(getShapeStr(layer.outShape))
            lineStr += '%-7s  '%(Layer.activationInfo[layer.activation][1])
            
            postActStrs = [ pa.getShortDesc() for pa in layer.postActivations ]
            if len('->'.join(postActStrs))>15:
                postActStrs = [ pa.name for pa in layer.postActivations ]
            lineStr += '%-15s  '%('->'.join(postActStrs))

            numParam = layer.getNumParams()
            lineStr += '%-11s'%('{:,}'.format(numParam))
            print(lineStr)

            totalParams += numParam

        #      123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345
        print('---------------------------------------------------------------------------------------------------------')
        print('                                                                  Total Number of parameters: %-11s'%('{:,}'.format(totalParams)))

    # ******************************************************************************************************************
    def getNumParams(self):
        numParam = 0
        for layer in self.layers: numParam += layer.getNumParams()
        return numParam

    # ******************************************************************************************************************
    def getAllNetParams(self, tfSession):
        # Returns a list of NetParam objects (with np content) for all parameters in all layers
        netParams = []
        for layer in self.layers:    netParams += NetParam.toNp(layer.netParams, tfSession)
        return netParams

# **********************************************************************************************************************
class Layer(object):
    layerClasses = {}
    numCustomLayers = 0
    activationInfo = {
                        'none': (None, 'None', 0),
                        'relu': (tf.nn.relu, 'ReLU', 1),
                        'selu': (tf.nn.selu, 'SeLU', 2),
                        'tanh': (tf.nn.tanh, 'Tanh', 3),
                        'sig': (tf.nn.sigmoid, 'Sigmoid', 4),
                        'soft': (tf.nn.softmax, 'Softmax', 5),
                        'gelu': (gelu, 'GELU', 6),
                     }
    name = "UNKNOWN"
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        self.layers = layers
        self.model = layers.model
        self.index = layerIndex
        self.inNetmark = None   # An integer number. If not None, input comes from the layer layers.netmarks[inNetmark]
        self.outNetmark = None  # An integer number. If not None, output goes to layers.netmarks[outNetmark]
        self.stage = 0
        self.scope = scope
        self.parent = parent
        if self.parent is None: self.scope += '_' + self.name    # Append layer name to the scope for 1 level
        self.activation = actStr
        self.postActivations = []
        self.isBlock = False
        
        # Shapes of input and output to/from layers:
        #    Conv. Models: height x width x channels
        #    BERT Models: seqLen x outSize
        #    FC Models: outSize
        self.inShape = None         # Shape of input to this layer (Batch dim not included)
        self.outShape = None        # Shape of output of this layer (Batch dim not included)
        
        self.isInput = False
        self.isOutput = False
        self.supportsEval = False   # Currently only "REG" layer supports this
        
        self.layerOuts = {'training':[], 'inference':[]}
        self.l2Loss = None
        
        self.netParams = None
        argVals = {argName: argDefault for argName,_,argDefault in self.argsDic.values() }
        self.updateArgVals(argVals, argsInfo)

        self.__dict__.update( argVals )

    # ******************************************************************************************************************
    def __repr__(self):
        """
        __repr__
        Returns a text string briefly describing this instance of "Layer".
        """
        retStr = '\n%s Instance:'%(self.__class__.__name__)
        retStr += '\n    %s: %s'%('name', self.name)
        retStr += '\n    %s: %s'%('scope', self.scope)
        retStr += '\n    %s: %s'%('index', self.index)
        for arg in self.__dict__:
            val = self.__dict__[arg]
            if arg in ['name', 'scope', 'index', 'argsDic', 'model']:   continue
            if val is None:                                             continue
            try:
                if arg == 'pathsLayers':    retStr += '\n    %s: %s'%(arg, str([len(pathLayer) for pathLayer in val]))
                elif arg == 'parent':       retStr += '\n    %s: %s'%(arg, str(val.scope))
                elif arg == 'block':        retStr += '\n    %s: %s'%(arg, str(val.name))
                else:                       retStr += '\n    %s: %s'%(arg, str(val))
            except:
                print("Failed printing %s!"%(arg))
                
        return retStr

    # ******************************************************************************************************************
    def getShortDesc(self):
        return ''

    # ******************************************************************************************************************
    def getInputStr(self):
        return 'Tensors of shape %s.'%(str(self.inShape))

    # ******************************************************************************************************************
    def getOutputStr(self):
        return 'Tensors of shape %s.'%(str(self.outShape))

    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        return []

    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        raise NotImplementedError("%s: L2 Regularization is not supported for \"%s\" layers!"%(self.name, self.scope))

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.netParams = []
        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = input
        return input, None
    
    # ******************************************************************************************************************
    def isVectorIn(self):
        return (len(self.inShape) == 1)
    
    # ******************************************************************************************************************
    def isDecomposed(self):
        return (self.__dict__.get('rank',0) > 0)

    # ******************************************************************************************************************
    def get4dPadding(self):
        padding4d = [[0,0]] + self.padding + [[0,0]]  # self.padding is [[top, bottom], [left,right]]
        return padding4d
    
    # ******************************************************************************************************************
    @property
    def nextLayer(self):
        if self.isOutput:               return None
        if self.parent is not None:     return self.parent.nextLayer
        return self.layers[self.index+1]

    # ******************************************************************************************************************
    @property
    def prevLayer(self):
        if self.isInput:                return None
        if self.parent is not None:     return self.parent.prevLayer
        return self.layers[self.index-1]

    # ******************************************************************************************************************
    def getInput(self, lastLayerOut, isTraining):
        if self.inNetmark is None: return lastLayerOut
        outKey = 'training' if isTraining else 'inference'
        return self.layers.netmarks[ self.inNetmark ].layerOuts[outKey][-1]
        
    # ******************************************************************************************************************
    def deepScope(self, postfix=None, sep='/'):
        deepStr = self.scope + sep
        if postfix is not None:     deepStr += postfix + sep
        if self.parent is not None: deepStr = self.parent.deepScope(sep=sep) + deepStr
        return deepStr

    # ******************************************************************************************************************
    @classmethod
    def getArgValue(cls, argType, argValStr):
        def getPairOfInts():
            if 'x' in argValStr:
                pairInfo = argValStr.split('x')
                return (int(pairInfo[0]), int(pairInfo[1]))
            return (int(argValStr), int(argValStr))

        def getListOfType():
            listStrs = argValStr.split('/')
            return [ Layer.getArgValue(argType, listStr) for listStr in listStrs ]

        def getPadding():
            if argValStr == 's':    return 'same'
            if argValStr == 'v':    return 'valid'
            if 'x' in argValStr:
                paddingNumStrs = argValStr.split('x')
                if len(paddingNumStrs)==2:                      # leftRight x topBottom
                    left = right = int(paddingNumStrs[0])
                    top = bottom = int(paddingNumStrs[1])
                elif len(paddingNumStrs)==4:                    # left x right x top x bottom
                    left = int(paddingNumStrs[0])
                    right = int(paddingNumStrs[1])
                    top = int(paddingNumStrs[2])
                    bottom = int(paddingNumStrs[3])
                else:
                    raise ValueError('Padding must be 1, 2 or 4 numbers "%s"!'%(argValStr))
            else:
                left = right = top = bottom = int(argValStr)
            return [ [top, bottom], [left, right] ]

        def getDecType():
            if argValStr == 'r':    return 'lr'
            elif argValStr == 'dr':  return 'ldr'
            else:
                raise ValueError('Unknown decomposition type "%s"!'%(argValStr))

        def getFmt():
            if argValStr == 's':    return 'ssd'
            else:
                raise ValueError('Unknown Feature Map Type "%s"!'%(argValStr))

        def getEmb():
            if argValStr == 'bert':     return 'bert'
            if argValStr == 'sig':      return 'sig'
            raise ValueError('Unknown Feature Map Type "%s"!'%(argValStr))

        if argType == 'i':      return int(argValStr)
        if argType == 'u':      return int(argValStr)
        if argType == 'ixi':    return getPairOfInts()
        if argType == 'uxu':    return getPairOfInts()
        if argType == 'f':      return float(argValStr)
        if argType == 'p':      return getPadding()
        if argType == 'dec':    return getDecType()
        if argType == 'b':      return int(argValStr)!=0
        if argType == 'fmt':    return getFmt()
        if argType == 'embt':   return getEmb()

        if '*' in argType:
            argType, nStr = argType.split('*')
            argVal = getListOfType()
            if nStr == '?': return argVal
            if len(argVal) != int(nStr):
                raise ValueError("List count mismatch for '%s' (%s values expected)"%(argValStr, nStr))
            return argVal
        raise ValueError("Unknown argument type '%s'!"%(argType))

    # ******************************************************************************************************************
    @property
    def npNetParams(self):
        return NetParam.toNp(self.netParams, self.model.session)

    # ******************************************************************************************************************
    @property
    def netParamValues(self):
        return NetParam.toNpValues(self.netParams, self.model.session)

    # ******************************************************************************************************************
    def getNpParams(self):
        return NetParam.toNp(self.netParams, self.model.session)

    # ******************************************************************************************************************
    def makeNetParam(self, name, initVal, trainable=True):
        if initVal.__class__.__name__ == 'NetParam':
            # This is when loading a model from a file. "initVal" is an "NP" NetParam object.
            # We ask initVal to create the "TF" NetParam for us.
            tfNetParam = initVal.makeTf(name, trainable)
        else:
            # This is when initializing with Random or constant values.
            tfInitValue = tf1.get_variable(initializer=initVal, name=name, trainable=trainable)
            tfNetParam = NetParam("TF", tfInitValue, None, None, trainable, name)
            tfNetParam.setSizeInfo( shape=tuple( x for x in initVal.shape ) )
            
        self.netParams += [ tfNetParam ]
        return tfNetParam.tfValue

    # ******************************************************************************************************************
    def getNumParams(self):
        assert(self.netParams is not None)
        return sum( netParam.numParams for netParam in self.netParams )

    # ******************************************************************************************************************
    def updateArgVals(self, argVals, argsInfo):
        for argInfo in argsInfo:
            if argInfo == '':   continue
            argKey = argInfo[0]
            argValStr = argInfo[1:]
            if argKey not in self.argsDic:
                print("%s: Ignoring unknown field '%s'!"%(self.scope, argInfo))
                continue
            
            argName, argType, _ = self.argsDic[argKey]
            if argValStr[0] == '%':
                assert self.parent is not None, "%s: %% sign is only allowed inside block definition!"%(self.scope)
                argVals[argName] = self.parent.getArgValueByKey(argValStr[1:])
            else:
                argVals[argName] = Layer.getArgValue(argType, argValStr)

        for argName in argVals:
            if argVals[argName] is None:
                raise ValueError("%s: The value of '%s' not specified!"%(self.scope, argName))

    # ******************************************************************************************************************
    def addPostActivations(self, postActivationStrs):
        for postActivationStr in postActivationStrs:
            paInfo = postActivationStr.split('_')
            paName = paInfo[0]
            paArgs = [] if len(paInfo)==1 else paInfo[1:]
            self.postActivations += [ PostActivation.createInstance(self, paName, paArgs) ]

    # ******************************************************************************************************************
    def getFeatureMap(self):
        for pa in self.postActivations:
            if pa.name == 'FM':
                return pa
        return None

    # ******************************************************************************************************************
    @classmethod
    def getArgStr(cls, argVal, argType):
        if argType == 'i':      return str(argVal)
        if argType == 'u':      return str(argVal)
        if argType == 'ixi':    return '%dx%d'%(argVal) if argVal[0]!=argVal[1] else str(argVal[0])
        if argType == 'uxu':    return '%dx%d'%(argVal) if argVal[0]!=argVal[1] else str(argVal[0])
        if argType == 'f':      return str(argVal)
        if argType == 'p':
            if argVal == 'same':        return 's'
            if argVal == 'valid':       return 'v'
            if (argVal[0][0] == argVal[0][1]) and (argVal[1][0] == argVal[1][1]):
                if argVal[0][0] == argVal[1][0]:    return str(argVal[0][0])
                return '%dx%d'%(argVal[1][0], argVal[0][0])
            return '%dx%dx%dx%d'%(argVal[1][0], argVal[1][1], argVal[0][0], argVal[0][1])
        if argType == 'dec':
            if argVal == 'none':        return 'n'
            elif argVal == 'lr':        return 'r'
            elif argVal == 'ldr':       return 'd'
        if argType == 'fmt':
            if argVal == 'ssd':         return 's'
        if argType == 'embt':
            if argVal == 'bert':        return 'bert'
            if argVal == 'sig':         return 'sig'
        if '*' in argType:
            listItemType, nStr = argType.split('*')
            listStrs = [ Layer.getArgStr(a, listItemType) for a in argVal ]
            return '/'.join(listStrs)

    # ******************************************************************************************************************
    def getLayerStr(self):
        layerStrs = []
        layerInfoStrs = [self.name]
        for argKey in self.argsDic:
            argName, argType, argDefault = self.argsDic[argKey]
            argVal = self.__dict__[argName]
            if argVal == argDefault:    continue
            layerInfoStrs += [ '%s%s'%(argKey.upper(), Layer.getArgStr(argVal, argType)) ]

        if len(self.postActivations) == 0:
            if self.activation == DEFAULT_ACTIVATION:
                return '_'.join(layerInfoStrs)

        actStr = Layer.activationInfo[self.activation][1].replace('Sigmoid','SIG').replace('Softmax','SOFT')
        layerStrs = ['_'.join(layerInfoStrs), actStr ] + [ pa.getLayerStr() for pa in self.postActivations ]
        
        layerStr = ':'.join(layerStrs)
        return layerStr
    
    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        raise NotImplementedError("'buildOnnx' function is not implemented for '%s' layer '%s'!"%(self.name, self.scope))

    # ******************************************************************************************************************
    def coreMlBuild(self, builder, inputName, lastOpName=None, netOutName=None):
        raise NotImplementedError("'coreMlBuild' function is not implemented for '%s' layer '%s'!"%(self.name, self.scope))
        
    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        raise NotImplementedError("'buildTf' function is not implemented for '%s' layer '%s'!"%(self.name, self.scope))

    # ******************************************************************************************************************
    def buildActivation(self, outputs, isTraining):
        tfName = None
        activationFunction = Layer.activationInfo[self.activation][0]
        tfName = Layer.activationInfo[self.activation][1]
        if activationFunction is None:  outputs += [ outputs[-1] ]
        else:                           outputs += [ activationFunction(outputs[-1], name=tfName) ]

    # ******************************************************************************************************************
    def buildDropout(self, input, dropRate):
        if self.model.dropRate == 1:        return None    # Dropout globally disabled
        if dropRate==1:     # Drop Rate not specified in layerInfo. Use global rate:
            dropRate = self.model.dropRate
        if dropRate<=0.0 or dropRate>=1.0:  return None
        try:
            return tf.nn.dropout(input, rate=dropRate, seed=SEED, name='Dropout')
        except:
            return tf.nn.dropout(input, keep_prob=1.0-dropRate, seed=SEED, name='Dropout')

    # ******************************************************************************************************************
    def buildOnnxActivation(self, onnxBuilder, inputName):
        outputName = inputName
        if self.activation in ['relu', 'tanh', 'sig', 'soft']:
            actName = {"relu":'Relu' , "tanh":'Tanh', "sig":'Sigmoid', "soft":'Softmax'}
            s = self.deepScope(actName[self.activation])
            outputName = s[:-1]
            onnxBuilder.addNode(actName[self.activation], [ inputName ], [outputName], outputName)

        elif self.activation == 'gelu':
            s = self.deepScope('GeLU')
            outputName = s[:-1]
            onnxBuilder.addParam(s+'1', 'float', [], [1.0])
            onnxBuilder.addParam(s+'2', 'float', [], [2.0])
            onnxBuilder.addParam(s+'3', 'float', [], [3.0])
            onnxBuilder.addParam(s+'0.04', 'float', [], [0.044715])
            onnxBuilder.addParam(s+'sqrt(2/pi)', 'float', [], [np.sqrt(2/np.pi)])

            onnxBuilder.addNode('Pow', [inputName, s+'3'], [s+'x3'], s+'Pow')
            onnxBuilder.addNode('Mul', [s+'x3', s+'0.04'], [s+'0.04*x3'], s+'Mul1')
            onnxBuilder.addNode('Add', [inputName, s+'0.04*x3'], [s+'x+0.04*x3'], s+'Add1')
            onnxBuilder.addNode('Mul', [s+'sqrt(2/pi)', s+'x+0.04*x3'], [s+'insideTanh'], s+'Mul2')
            onnxBuilder.addNode('Tanh', [s+'insideTanh'], [s+'tanh'], s+'Tanh')
            onnxBuilder.addNode('Add', [s+'1', s+'tanh'], [s+'1+tanh'], s+'Add2')
            onnxBuilder.addNode('Div', [s+'1+tanh', s+'2'], [s+'cdf'], s+'Div')
            onnxBuilder.addNode('Mul', [inputName, s+'cdf'], [outputName], s+'Mul3')

        elif self.activation == 'none':
            outputName = inputName
            
        else:
            raise NotImplementedError("'%s' Activation not implemented!"%(self.activation))
        
        return outputName

    # ******************************************************************************************************************
    def buildCmlActivation(self, cmlBuilder, inputName):
        if self.activation in ['relu', 'tanh', 'sig']:
            actName = {"relu":'RELU' , "tanh":'TANH', "sig":'SIGMOID'}
            s = self.deepScope(actName[self.activation])
            outputName = s[:-1]
            cmlBuilder.add_activation(name =           outputName,
                                      non_linearity =  actName[self.activation],
                                      input_name =     inputName,
                                      output_name =    outputName)
            
        elif self.activation == 'soft':
            s = self.deepScope('SOFTMAX')
            outputName = s[:-1]
            cmlBuilder.add_softmax(name =              outputName,
                                   input_name =        inputName,
                                   output_name =       outputName)

        elif self.activation == 'gelu':
            s = self.deepScope('GELU')
            outputName = s[:-1]
            cmlBuilder.add_gelu(outputName, inputName, outputName, mode='TANH_APPROXIMATION')

        elif self.activation == 'none':
            outputName = inputName
        else:
            raise NotImplementedError("'%s' Activation not implemented!"%(self.activation))
        
        return outputName

    # ******************************************************************************************************************
    def buildTfActivation(self, tfBuilder):
        if self.activation in ['gelu']:
            if not tfBuilder.methodDefined('gelu'):
                tfBuilder.defineMethod('gelu')
                tfBuilder.addMethod(("def gelu(self, x, name):",
                                     "    with tf.name_scope(name):",
                                     "        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))",
                                     "        output = x * cdf",
                                     "    return output",
                                     ""))
        
        if self.activation != 'none':
            tfName = Layer.activationInfo[self.activation][1]
            actFuncName = { 'relu': 'tf.nn.relu', 'selu': 'tf.nn.selu', 'tanh': 'tf.nn.tanh', 'sig': 'tf.nn.sigmoid',
                            'soft': 'tf.nn.softmax', 'gelu': 'gelu' }
            tfBuilder.addToGraph("out = %s(out, name='%s')"%(actFuncName[self.activation], tfName))

    # ******************************************************************************************************************
    def buildPostActivation(self, outputs, isTraining):
        for p, pa in enumerate(self.postActivations):
            if pa.name in ['MP','AP']:
                kernelX, kernelY = pa.kernel
                strideX, strideY = pa.stride
                poolFunc = tf.nn.max_pool2d if pa.name=='MP' else tf.nn.avg_pool

                if pa.padding not in ['valid', 'same']:
                    padding4d = [[0,0]] + pa.padding + [[0,0]]  # pa.padding is [[top, bottom], [left,right]]
                    outputs += [ poolFunc(tf.pad(outputs[-1], padding4d, name='Pad'),
                                          ksize=[1, kernelY, kernelX, 1],
                                          strides=[1, strideY, strideX, 1],
                                          padding='VALID') ]
                else:
                    outputs += [ poolFunc(outputs[-1],
                                          ksize=[1, kernelY, kernelX, 1],
                                          strides=[1, strideY, strideX, 1],
                                          padding=pa.padding.upper()) ]
            
            elif pa.name=='AAP':
                # TODO: Implement Average Pooling
                raise NotImplementedError("%s: Average Pooling not implemented yet!"%(self.scope))

            elif pa.name=='UP':
                scaleX, scaleY = pa.scale
                outputs += [ tf1.image.resize_nearest_neighbor(outputs[-1], (pa.outY, pa.outX), name='UpSampling') ]

            elif pa.name=='DO':
                if not isTraining:      continue    # No Dropout when not training
                dropOutput = self.buildDropout(outputs[-1], pa.dropRate)
                if dropOutput is not None:      outputs += [ dropOutput ]

            elif pa.name=='GAP':
                outputs += [ tf.reduce_mean(outputs[-1], [1,2], keepdims=False) ]

            elif pa.name=='CLP':
                if pa.normVal != np.inf:
                    outputs += [ tf.clip_by_norm(outputs[-1], pa.normVal, name='CLIP') ]
                else:
                    outputs += [ tf.clip_by_value(outputs[-1], pa.loVal, pa.hiVal, name='CLIP') ]

            elif pa.name=='L2R':
                if not isTraining:              continue
                if self.model.regFactor == 0:   continue
                factor = pa.factor if pa.factor != 1.0 else self.model.regFactor
                self.makeL2Loss(factor)

            elif pa.name=='FM':
                self.layers.paFms[ pa.fmIndex ] = (pa, outputs[-1])
                
            elif pa.name=='TP':
                outputs += [ tf.reshape(outputs[-1][:,0:pa.numVectors,:], [-1, pa.numVectors*self.outSize]) ]
                
            elif pa.name=='ADD':
                outKey = 'training' if isTraining else 'inference'
                layerOutputs = []
                if len(pa.netmarks) == 1:
                    # For 1 addition we support broadcasting addition
                    layerOut = self.layers.netmarks[ pa.netmarks[0] ].layerOuts[outKey][-1]
                    if type(layerOut)==int: layerOut = self.layers.netmarks[ layerOut ].layerOuts[outKey][-1]

                    diff = len(layerOut.shape) - len(outputs[-1].shape)
                    if diff < 0:
                        newShape = layerOut.shape.as_list() + [1]*( -diff )
                        outputs += [ tf.reshape(layerOut, [-1 if x is None else x for x in newShape]) + outputs[-1] ]
                    elif diff > 0:
                        newShape = outputs[-1].shape.as_list() + [1]*( diff )
                        outputs += [ tf.reshape(outputs[-1], [-1 if x is None else x for x in newShape]) + layerOut ]
                    else:
                        outputs += [ outputs[-1] + layerOut ]
                else:
                    for l in pa.netmarks:
                        layerOut = self.layers.netmarks[ l ].layerOuts[outKey][-1]
                        if type(layerOut)==int: layerOut = self.layers.netmarks[ layerOut ].layerOuts[outKey][-1]
                        layerOutputs += [ layerOut ]
                    layerOutputs += [ outputs[-1] ]
                    outputs += [ tf.add_n(layerOutputs, name='Add') ]
                
            elif pa.name=='SEL':
                outKey = 'training' if isTraining else 'inference'
                layerOutputs = []
                for l in pa.netmarks:
                    layerOut = self.layers.netmarks[ l ].layerOuts[outKey][-1]
                    if type(layerOut)==int: layerOut = self.layers.netmarks[ layerOut ].layerOuts[outKey][-1]
                    layerOutputs += [ layerOut ]

                if len(layerOutputs)==2:
                    outputs += [ tf.where_v2(tf.less(outputs[-1],0.5), layerOutputs[0], layerOutputs[1]) ]
                else:
                    layerOutputs = tf.stack(layerOutputs)
                    indexes0 = tf.cast( tf.argmax(outputs[-1], axis=1), dtype=tf.int32)
                    indexes1 = tf.range(tf.shape(indexes0)[0], dtype=tf.int32)
                    indexes2d = tf.stack([indexes0,indexes1], axis=1)
                    selOut = tf.gather_nd(layerOutputs, indexes2d)
                    outputs += [ selOut ]

            elif pa.name=='WSUM':
                outKey = 'training' if isTraining else 'inference'
                layerOutputs = []
                for l in pa.netmarks:
                    layerOut = self.layers.netmarks[ l ].layerOuts[outKey][-1]
                    if type(layerOut)==int: layerOut = self.layers.netmarks[ layerOut ].layerOuts[outKey][-1]
                    layerOutputs += [ layerOut ]
                    
                if len(layerOutputs)==2: outputs += [ outputs[-1]*layerOutputs[1] + (1.0-outputs[-1])*layerOutputs[0] ]
                else:                    outputs += [ tf.reduce_sum(tf.stack(layerOutputs, axis=1) * outputs[-1], axis=1) ]

            elif pa.name=='TUP':
                outKey = 'training' if isTraining else 'inference'
                layerOutputs = []
                for l in pa.netmarks:
                    layerOut = self.layers.netmarks[ l ].layerOuts[outKey][-1]
                    if type(layerOut)==int: layerOut = self.layers.netmarks[ layerOut ].layerOuts[outKey][-1]
                    layerOutputs += [ layerOut ]
                outputs += [ tuple([outputs[-1]] + layerOutputs) ]

    # ******************************************************************************************************************
    def buildOnnxPostActivation(self, onnxBuilder, inputName):
        outputName = inputName
        for p, pa in enumerate(self.postActivations):
            s = self.deepScope(pa.name)
            if pa.name in ['MP','AP']:
                kernelX, kernelY = pa.kernel
                strideX, strideY = pa.stride
                
                onnxOp = 'MaxPool' if pa.name == 'MP' else 'AveragePool'
                outputName = s[:-1]
                if (pa.padding == 'same') or (pa.padding == 'valid'):
                    onnxBuilder.addNode(onnxOp, [inputName], [outputName], s+onnxOp,
                                        kernel_shape=[kernelY, kernelX], strides=[strideY, strideX],
                                        auto_pad={'same':"SAME_UPPER", 'valid':"VALID"}[pa.padding])
                else:
                    # self.padding is: [ [top, bottom], [left, right] ]
                    # onnx padding must be: [ top, left, bottom, right ]
                    onnxPaddingValues = [ pa.padding[0][0], pa.padding[1][0], pa.padding[0][1], pa.padding[1][1] ]
                    onnxBuilder.addNode(onnxOp, [inputName], [outputName], s+onnxOp,
                                        kernel_shape=[kernelY, kernelX], strides=[strideY, strideX],
                                        auto_pad='NOTSET', pads=onnxPaddingValues)
                inputName = outputName

            elif pa.name=='UP':
                scaleX, scaleY = pa.scale
                outputName = s[:-1]
                onnxBuilder.addParam(s+'Scales', 'float', [4], np.float32([1,1,scaleY, scaleX]))
                onnxBuilder.addNode('Resize', [inputName, '', s+'Scales'], [outputName], s+'Resize', mode='nearest')
                inputName = outputName

            elif pa.name=='GAP':
                outputName = s[:-1]
                onnxBuilder.addNode('GlobalAveragePool', [inputName], [outputName], s+'GlobalAveragePool')
                inputName = outputName

            elif pa.name=='CLP':
                if pa.normVal != np.inf:
                    raise NotImplementedError("Clip by Norm not implemented!")
                
                # Note: The ONNX CLIP operator has a bug. It crashes with negative min value if CLIP is following
                # a ReLU!
                outputName = s[:-1]
                minOutName = outputName if pa.hiVal == np.inf else s+'ClipMin'
                if pa.loVal != -np.inf:
                    minName = s+'MinVal'
                    onnxBuilder.addParam(minName, 'float', [], [pa.loVal])
                    onnxBuilder.addNode('Max', [inputName, minName], [minOutName], s+'/Max')

                if pa.hiVal != np.inf:
                    maxName = s+'MaxVal'
                    onnxBuilder.addParam(maxName, 'float', [], [pa.hiVal])
                    onnxBuilder.addNode('Min', [inputName, maxName], [outputName], s+'Min')
                    
                inputName = outputName
                    
            elif pa.name=='FM':
                pa.inputName = inputName

            elif pa.name=='TP':
                raise NotImplementedError("Transformers Pooling not implemented yet!")
                
        return outputName

    # ******************************************************************************************************************
    def buildCmlPostActivation(self, cmlBuilder, inputName):
        for p, pa in enumerate(self.postActivations):
            s = self.deepScope(pa.name)
            if pa.name in ['MP','AP']:
                outputName = s[:-1]
                kernelX, kernelY = pa.kernel
                strideX, strideY = pa.stride
                cmlBuilder.add_pooling(name =              outputName,
                                       height =            kernelY,
                                       width =             kernelX,
                                       stride_height =     strideY,
                                       stride_width =      strideX,
                                       layer_type =        'MAX' if pa.name=='MP' else 'AVERAGE',
                                       padding_type =      'SAME' if pa.padding=='same' else 'VALID',
                                       input_name =        inputName,
                                       output_name =       outputName,
                                       exclude_pad_area =  True,
                                       is_global =         False,
                                       padding_top =       0 if pa.padding in ['valid', 'same'] else pa.padding[0][0],
                                       padding_bottom =    0 if pa.padding in ['valid', 'same'] else pa.padding[0][1],
                                       padding_left =      0 if pa.padding in ['valid', 'same'] else pa.padding[1][0],
                                       padding_right =     0 if pa.padding in ['valid', 'same'] else pa.padding[1][1],
                                       same_padding_asymmetry_mode = 'BOTTOM_RIGHT_HEAVY')
                inputName = outputName

            elif pa.name=='UP':
                outputName = s[:-1]
                scaleX, scaleY = pa.scale
                cmlBuilder.add_upsample(name =             outputName,
                                     scaling_factor_h = scaleY,
                                     scaling_factor_w = scaleX,
                                     input_name =       inputName,
                                     output_name =      outputName,
                                     mode =             'NN')
                inputName = outputName
                
            elif pa.name=='GAP':
                outputName = s[:-1]
                cmlBuilder.add_pooling(name =              outputName,
                                       height =            0,
                                       width =             0,
                                       stride_height =     0,
                                       stride_width =      0,
                                       layer_type =        'AVERAGE',
                                       padding_type =      'VALID',
                                       input_name =        inputName,
                                       output_name =       outputName,
                                       exclude_pad_area =  False,
                                       is_global =         True)
                inputName = outputName

            elif pa.name=='CLP':
                outputName = s[:-1]
                if pa.normVal != np.inf:
                    raise NotImplementedError("Clip by Norm not implemented!")
                else:
                    cmlBuilder.add_clip(name =         outputName,
                                        input_name =   inputName,
                                        output_name =  outputName,
                                        min_value =    max(np.finfo('f').min, pa.loVal),
                                        max_value =    min(np.finfo('f').max, pa.hiVal))
                inputName = outputName
        
            elif pa.name=='FM':
                pa.inputName = inputName

        return inputName

    # ******************************************************************************************************************
    def buildTfPostActivation(self, tfBuilder):
        for p, pa in enumerate(self.postActivations):
            if pa.name in ['MP','AP']:
                kernelX, kernelY = pa.kernel
                strideX, strideY = pa.stride
                poolFunc = 'tf.nn.max_pool2d' if pa.name=='MP' else 'tf.nn.avg_pool'

                if pa.padding not in ['valid', 'same']:
                    padding4d = [[0,0]] + pa.padding + [[0,0]]  # pa.padding is [[top, bottom], [left,right]]
                    
                    tfBuilder.addToGraph("out = %s(%s, ksize=[1, %d, %d, 1], strides=[1, %d, %d, 1], padding='VALID')",
                                         (poolFunc, 'tf.pad(out, '+str(padding4d)+', name="Pad")',
                                          kernelY, kernelX, strideY, strideX))
                else:
                    tfBuilder.addToGraph("out = %s(out, ksize=[1, %d, %d, 1], strides=[1, %d, %d, 1], padding='%s')",
                                         (poolFunc, kernelY, kernelX, strideY, strideX, pa.padding.upper()))

            elif pa.name=='AAP':
                # TODO: Implement Average Pooling
                raise NotImplementedError("%s: Average Pooling not implemented yet!"%(self.scope))
                
            elif pa.name=='UP':
                tfBuilder.addToGraph("out = tf1.image.resize_nearest_neighbor(out, (%d,%d), name='UpSampling')",
                                     (pa.outY, pa.outX))

            elif pa.name=='DO':
                if self.model.dropRate == 1:        continue    # Dropout globally disabled
                dropRate = pa.dropRate
                if dropRate==1:             # Drop Rate not specified in layerInfo. Use global rate:
                    dropRate = self.model.dropRate
                if dropRate<=0.0 or dropRate>=1.0:  continue
                tfBuilder.addToGraph(("if isTraining:",
                                      "    try:    out = tf.nn.dropout(out, rate=%s, seed=SEED, name='Dropout')",
                                      "    except: out = tf.nn.dropout(out, keep_prob=%s, seed=SEED, name='Dropout')"),
                                     (str(dropRate), str(1.0-dropRate)))

            elif pa.name=='GAP':
                tfBuilder.addToGraph("out = tf.reduce_mean(out, [1,2], keepdims=False)")

            elif pa.name=='CLP':
                if pa.normVal != np.inf:
                    tfBuilder.addToGraph("out = tf.clip_by_norm(out, %f, name='CLIP')"%(pa.normVal))
                else:
                    lo = "-np.inf" if pa.loVal == -np.inf else str(pa.loVal)
                    hi = "np.inf" if pa.hiVal == np.inf else str(pa.hiVal)
                    tfBuilder.addToGraph("out = tf.clip_by_value(out, %s, %s, name='CLIP')"%(lo, hi))

            elif pa.name=='FM':
                tfBuilder.addToGraph("self.featureMaps += [ out ]")

            elif pa.name=='TP':
                tfBuilder.addToGraph("out = tf.reshape(out[:,0:%d,:], [-1, %d])",
                                     (pa.numVectors, pa.numVectors*self.outSize))


    # ******************************************************************************************************************
    def decomposeMatrix(self, w, rankInfo):
        u, s, vT = np.linalg.svd(w, full_matrices=True)
            
        def getGH(r):
            ssr = np.sqrt(s[:r])
            return u[:,:r]*ssr, np.diag(ssr).dot(vT[:r,:])
        
        def getMse(r):
            g, h = getGH(r)
            return np.square(g.dot(h)-w).mean()
        
        if type(rankInfo) == int:
            g, h = getGH(rankInfo)
            mse = np.square(g.dot(h)-w).mean()
            return rankInfo, g, h, mse
            
        mseUB = float(rankInfo)
        rankHi = int(np.prod(w.shape)/sum(w.shape))
        rankLo = max(rankHi//10,2)
        mseLo = getMse(rankLo)
        while mseLo<mseUB:
            if rankLo == 2: break
            rankLo = max(rankLo//2,2)
            mseLo = getMse(rankLo)

        mseHi = getMse(rankHi)
        if mseHi>mseUB:
            return rankHi, None, None, mseHi    # Cannot Decompose

        if mseLo<mseUB:
            rank = rankLo
        else:
            while (rankHi-rankLo)>=2:
                rank = int(rankLo + (rankHi-rankLo)*(mseUB-mseLo)/(mseHi-mseLo))
                if rank>=rankHi:     rank = rankHi-1
                if rank<=rankLo:     rank = rankLo+1
                newMse = getMse(rank)
                if newMse>mseUB:    rankLo, mseLo = rank, newMse
                elif newMse<mseUB:  rankHi, mseHi = rank, newMse
                else:
                    break

            if mseLo<=mseUB:        rank = rankLo
            elif mseHi<=mseUB:      rank = rankHi
            else:
                return rankHi, None, None, mseHi    # Cannot Decompose

        # We want the rank to be a multiple of 8 to make the matrix multiplications fast.
        RANK_FACTOR = 8 # Making sure rank is a multiple of this
        if rank<=2:     rank=2
        elif rank<=4:   rank=4
        else:           rank = int(np.round(np.float32(rank)/RANK_FACTOR)*RANK_FACTOR)
        
        maxRank = int(w.size/sum(w.shape))
        if rank >= maxRank:     # We don't want to increase the tensor size!
            return rank, None, None, None    # Cannot Decompose
                        
        # Note: MSE may go above mseUB because of this, but we assume on average these changes
        # for different tensors cancel eachother.
        g, h = getGH(rank)
        mse = np.square(g.dot(h)-w).mean()
        
        return rank, g, h, mse

    # ******************************************************************************************************************
    def decompose(self, session, decInfo, decomposeDWCN=True):
        # params:   [w,b,...]
        # decInfo:  decType, rank, moreInfo
        #           decType:    'lr' or 'ldr'
        #           rank:       int->rank or
        #                       float->MSE Upper Bound => (Find best rank)
        #           moreInfo:   (e,f) (ignored for LR)
        assert (self.name in ['FC', 'CONV', 'DWCN']), "%s: Cannot decompose '%s' layers!"%(self.scope, self.name)
        assert (self.isDecomposed() == False), "%s: Layer already decomposed!"%(self.scope)
        
        decType, rank, decMode = decInfo
        params = NetParam.toNpValues(self.netParams, session)
        w, b = params[0], None
        if len(params)>1: b = params[1]
        
        if self.name == 'CONV':
            w = w.reshape((-1, w.shape[3]))
#            assert (decMode in [1,2]), "%s: Decomposition Mode %d not supported!"%(self.scope, decMode)
#            if decMode == 1:    w = w.reshape((-1, w.shape[3]))
#            else:               w = w.reshape((w.shape[0]*w.shape[2], w.shape[1]*w.shape[3]))
        elif self.name == 'DWCN':
            if decomposeDWCN == False:
                return (None, None, None, '%s => Skipping Depth-wise Convolution layer.'%(self.scope))
            elif (w.shape[0] == w.shape[1]) and (w.shape[0] in DwConvLayer.num2Factors) and (w.shape[2] in DwConvLayer.num2Factors[w.shape[0]]):
                w = w.reshape( DwConvLayer.num2Factors[w.shape[0]][w.shape[2]] )
            else:
                return (None, None, None, '%s => Cannot decompose tensor with shape: %s'%(self.scope, str(w.shape)))

        shapeStr = '' if self.name == 'FC' else 'Shape: %s, '%(str(w.shape))
        mseUB = float(rank)
        if decType == 'lr':
            rank, g, h, mse = self.decomposeMatrix(w, rank)
            if (g is None) or (h is None):
                if mse is None:
                    maxRank = int(np.prod(w.shape)/sum(w.shape))
                    return (None, None, None,
                        '%s => Cannot Decompose, %sRank(%d)>MaxRank(%d)'%(self.scope, shapeStr, rank, maxRank))
                        
                return (None, None, None,
                        '%s => Cannot Decompose, %sMSE(%d)=%f>%f'%(self.scope, shapeStr, rank, mse, mseUB))

            layerStrParts = self.getLayerStr().split(':')
            layerStrParts[0] = layerStrParts[0] + '_R%d'%(rank)
            newLayerStr = ':'.join(layerStrParts)
            
            numOldParams = np.prod(w.shape)
            numNewParams = rank*sum(w.shape)
            assert (numOldParams>numNewParams), \
                   "numOldParams(%d)>numNewParams(%d), rank:%d"%(numOldParams, numNewParams, rank) # Should never happen
            
            changeStr = 'Reduction: %.1f%%'%((numOldParams-numNewParams)*100.0/numOldParams)
            infoStr = '%s => LR(%d), MSE=%f, %sParams: %d->%d (%s)'%(self.scope, rank, mse, shapeStr,
                                                                     numOldParams, numNewParams, changeStr)
            
            if self.name == 'CONV':
                g = g.reshape((self.kernel[1], self.kernel[0], -1, rank))
                h = h.reshape(1, 1, rank, -1)

            if b is None:   return ([g, h], newLayerStr, numNewParams, infoStr)
            return ([g, h, b], newLayerStr, numNewParams + b.size, infoStr)

        if decType == 'ldr':
            e, f = moreInfo
            g,jH,e,f,rank,mse = ldr.getLdrGH(weights, e, f, rank, None, np.float32)

            layerStrParts = self.getLayerStr().split(':')
            layerStrParts[0] = layerStrParts[0] + '_Ldr_R%d'%(rank)
            if e != FcLayer.argsDic['e'][2]: layerStrParts[0] += '_E%f'%(e)
            if f != FcLayer.argsDic['e'][2]: layerStrParts[0] += '_F%f'%(f)
            newLayerStr = ':'.join(layerStrParts)

            numOldParams = np.prod(w.shape)
            numNewParams = rank*sum(w.shape)
            if numOldParams>numNewParams: changeStr = 'Change: -%.1f%%'%((numOldParams-numNewParams)*100.0/numOldParams)
            else:                         changeStr = 'Change: +%.1f%%'%((numNewParams-numOldParams)*100.0/numOldParams)
            infoStr = '%s => LDR, Rank %d, e,f: %d,%d MSE= %f, Params: %d (%s)'%(self.scope, rank, e,f,
                                                                                 mse, numNewParams, changeStr)

            if b is None:   return ([g, hT], numNewParams, numNewParams, infoStr)
            return ([g, jH, b], newLayerStr, numNewParams + b.size, infoStr)
    
    # ******************************************************************************************************************
    @classmethod
    def getByteListForArg(cls, argType, argVal):
        if argType == 'i':                      return wl.int2ByteList(argVal)
        if argType == 'u':                      return wl.uint2ByteList(argVal)
        if argType == 'b':                      return wl.uint2ByteList(1 if argVal else 0)
        if argType == 'ixi':                    return (wl.int2ByteList(argVal[0]) + wl.int2ByteList(argVal[1]))
        if argType == 'uxu':                    return (wl.uint2ByteList(argVal[0]) + wl.uint2ByteList(argVal[1]))
        if argType == 'f':                      return wl.shortFloat2ByteList(argVal)
        if argType == 'p':
            padding = argVal
            if padding == 'valid':                      paddingCode = 0
            elif padding == 'same':                     paddingCode = 1
            elif (padding[0][0] == padding[0][1]) and (padding[1][0] == padding[1][1]):
                if padding[0][0] == padding[1][0]:      paddingCode = 2     # 1 value for all padding
                else:                                   paddingCode = 3     # 2 values for x/y (Symmetric)
            else:                                       paddingCode = 4     # 4 values for all Different paddings
            byteList = wl.uint2ByteList(paddingCode)
            # padding is [[top, bottom], [left,right]]
            if paddingCode==2:
                # 1 value for all padding
                byteList += wl.uint2ByteList(padding[0][0])
            elif paddingCode==3:
                # 2 values for x/y (Symmetric)
                # Write order: leftRight , topBottom
                byteList += wl.uint2ByteList(padding[1][0])
                byteList += wl.uint2ByteList(padding[0][0])
            elif paddingCode==4:
                # 4 values for all Different paddings
                # Write order: left , right , top , bottom
                byteList += wl.uint2ByteList(padding[1][0])
                byteList += wl.uint2ByteList(padding[1][1])
                byteList += wl.uint2ByteList(padding[0][0])
                byteList += wl.uint2ByteList(padding[0][1])
            return byteList

        if argType == 'dec':
            if argVal == 'lr':      return wl.uint2ByteList(1)
            elif argVal == 'ldr':   return wl.uint2ByteList(2)
            assert False, "Invalid Decomposition Code! (%s)"%(argVal)

        if argType == 'fmt':
            if argVal == 'ssd':     return wl.uint2ByteList(1)
            assert False, "Invalid Feature Map Code! (%s)"%(argVal)

        if argType == 'embt':
            if argVal == 'bert':    return wl.uint2ByteList(1)
            if argVal == 'sig':     return wl.uint2ByteList(2)
            assert False, "Invalid Embedding Type Code! (%s)"%(argVal)

        if '*' in argType:
            argType, nStr = argType.split('*')
            byteList = []
            for a in argVal:
                byteList += Layer.getByteListForArg(argType, a)
            return byteList

        assert False, "Invalid argument type! (%s)"%(argType)

    # ******************************************************************************************************************
    def getByteList(self):
        if self.name.lower() not in Layer.layerClasses:
            # This must be a BlockInstance
            byteList = wl.uint2ByteList(0)
            byteList += wl.str2ByteList(self.name)
        else:
            typeId = Layer.layerClasses[self.name.lower()][1]
            byteList = wl.uint2ByteList(typeId)
        byteList += wl.uint2ByteList(self.index)
        byteList += wl.uint2ByteList(self.stage)
    
        # Now add all arguments in the argsDic
        for key in self.orderedKeys:
            (argName, argType, _) = self.argsDic[key]
            argVal = self.__dict__[argName]
            byteList += Layer.getByteListForArg(argType, argVal)
        
        # Adding Activation Function Info
        actId = Layer.activationInfo[self.activation][2]
        byteList += wl.uint2ByteList(actId)

        # Adding Post-Activation Information:
        byteList += wl.uint2ByteList(len(self.postActivations))
        for pa in self.postActivations:
            byteList += pa.getByteList()

        return byteList

    # ******************************************************************************************************************
    @classmethod
    def byteList2ArgValStr(cls, argType, byteList, offset=None):
        dataBytes = byteList if offset is None else byteList[offset:]

        b = 0
        argValStr = None
        if argType == 'i':          argVal,b = wl.byteList2Int(dataBytes, b);          argValStr = str(argVal)
        elif argType == 'u':        argVal,b = wl.byteList2Uint(dataBytes, b);         argValStr = str(argVal)
        elif argType == 'b':        argVal,b = wl.byteList2Uint(dataBytes, b);         argValStr = str(argVal)
        elif argType == 'f':        argVal,b = wl.byteList2ShortFloat(dataBytes, b);   argValStr = str(argVal)
        elif argType == 'ixi':
            argVal0,b = wl.byteList2Int(dataBytes, b)
            argVal1,b = wl.byteList2Int(dataBytes, b)
            argValStr = '%dx%d'%(argVal0,argVal1) if argVal0 != argVal1 else str(argVal0)
        
        elif argType == 'uxu':
            argVal0,b = wl.byteList2Uint(dataBytes, b)
            argVal1,b = wl.byteList2Uint(dataBytes, b)
            argValStr = '%dx%d'%(argVal0,argVal1) if argVal0 != argVal1 else str(argVal0)

        elif argType == 'p':
            paddingCode,b = wl.byteList2Uint(dataBytes, b)
            if paddingCode == 0:            argValStr = 'v'
            elif paddingCode == 1:          argValStr = 's'
            elif paddingCode == 2:
                # 1 value for all padding
                argVal,b = wl.byteList2Uint(dataBytes, b)
                argValStr = str(argVal)
            elif paddingCode == 3:
                # 2 values for x/y (Symmetric)
                # leftRight x topBottom
                pX,b = wl.byteList2Uint(dataBytes, b)
                pY,b = wl.byteList2Uint(dataBytes, b)
                argValStr = '%dx%d'%(pX,pY)
            elif paddingCode == 4:
                # 4 values for all Different paddings
                # left x right x top x bottom
                pLeft,b = wl.byteList2Uint(dataBytes, b)
                pRight,b = wl.byteList2Uint(dataBytes, b)
                pTop,b = wl.byteList2Uint(dataBytes, b)
                pBottom,b = wl.byteList2Uint(dataBytes, b)
                argValStr = '%dx%dx%dx%d'%(pLeft,pRight,pTop,pBottom)
            else:
                assert False, "Invalid Padding Code! (%d)"%(paddingCode)

        elif argType == 'dec':
            decCodeId,b = wl.byteList2Uint(dataBytes, b)
            if decCodeId == 1:      argValStr = 'r'
            elif decCodeId == 2:    argValStr = 'dr'
            else:                   assert False, "Invalid Decomposition Code Id! (%d)"%(decCodeId)
        
        elif argType == 'fmt':
            decCodeId,b = wl.byteList2Uint(dataBytes, b)
            if decCodeId == 1:      argValStr = 's'
            else:                   assert False, "Invalid Feature Map Code! (%d)"%(decCodeId)

        elif argType == 'embt':
            if decCodeId == 1:      argValStr = 'bert'
            elif decCodeId == 2:    argValStr = 'sig'
            else:                   assert False, "Invalid Embedding Type Code! (%s)"%(decCodeId)

        elif '*' in argType:
            listItemType, nStr = argType.split('*')
            n = int(nStr)
            argValStrs = []
            for a in range(n):
                argValStr,b = Layer.byteList2ArgValStr(listItemType, dataBytes, b)
                argValStrs += [ argValStr ]
            argValStr = '/'.join(argValStrs)

        else:
            assert False, "Invalid argType! (%s)"%(argType)

        if offset is None: return argValStr
        return argValStr, (offset+b)

    # ******************************************************************************************************************
    def checkShape(self, pName, initializerShape, expectedShape):
        assert initializerShape==expectedShape, ("%s: Shape mismatch when initializing \"%s\"!"%(self.scope, pName) +
                                                 "(Expected: %s, "%(str(expectedShape)) +
                                                 "Initializer: %s)"%(str(initializerShape)))

    # ******************************************************************************************************************
    def inferOut(self, session, feedDic, subLayer=-1, oneSample=False):
        # Note: All layer objects have the graph info related to the last tower only. So, this runs on
        # the last tower. There is no Parallelization. If the number of samples is large the GPU can run
        # out of memory.
        results = session.run(self.layerOuts['inference'][subLayer], feed_dict=feedDic)
        if oneSample:   results = results[0]
        return results

    # ******************************************************************************************************************
    def feed(self, data, towers=None):
        # Each input or output layer has a tuple called placeholders.
        # Tower has a tuple labelsPlaceholders for outputs, and samplesPlaceholders for input.
        # if there is more than one item in the placeholders, the data must also be a tuple
        # when samplesPlaceholders has more than one item, the data may have an extra item. The first
        # item in data should be ignored (this is usually the sample indexes)
        feedDic = {}
        if self.isInput:
            if type(data)==tuple:
                n = len(self.placeholders)
                samplesTuple = data[-n:]
            else:
                samplesTuple = (data,)
            assert len(self.placeholders) == len(samplesTuple)

            if towers is None:
                for samplesPlaceholder, samples in zip(self.placeholders,samplesTuple):
                    feedDic[ samplesPlaceholder ] = samples
            else:
                batchSize = len(samplesTuple[0])
                numTowers = len(towers)
                towerBatchSize = batchSize//numTowers
                if towerBatchSize == 0: return None
                for t,tower in enumerate(towers):
                    for samplesPlaceholder, samples in zip(tower.samplesPlaceholders,samplesTuple):
                        feedDic[ samplesPlaceholder ] = samples[t*towerBatchSize : (t+1)*towerBatchSize]

        elif self.isOutput:
            labelsTuple = data if type(data)==tuple else (data,)
            assert len(self.placeholders) == len(labelsTuple)

            if towers is None:
                for labelPlaceholder, labels in zip(self.placeholders,labelsTuple):
                    feedDic[ labelPlaceholder ] = labels
            else:
                batchSize = len(labelsTuple[0])
                numTowers = len(towers)
                towerBatchSize = batchSize//numTowers
                if towerBatchSize == 0: return None
                for t,tower in enumerate(towers):
                    for labelPlaceholder, labels in zip(tower.labelsPlaceholders,labelsTuple):
                        feedDic[ labelPlaceholder ] = labels[t*towerBatchSize : (t+1)*towerBatchSize]

        else:
            raise ValueError("'feed' can only be called on input or output layers! (not '%s' layers)"%(self.name))

        return feedDic

    # ******************************************************************************************************************
    def postProcessResults(self, rawResults, returnProbs):
        assert self.isOutput, "'postProcessResults' can only be called on output layers! (not '%s' layers)"%(self.name)
        return rawResults

# **********************************************************************************************************************
# MARK: ------------------------ Input Layers ------------------------
class ImageInLayer(Layer):
    argsDic = {
                'd': ('depth', 'u', 3),         # Options: 3 for RGB/GBR, 1 for Monochrome
                's': ('imgSize', 'uxu', None),  # imageSize is width x height. The Input Shape is: (height, width, depth)
              }
    orderedKeys = 'ts'
    name = 'IMG'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isInput = True

    # ******************************************************************************************************************
    def makePlaceholders(self):
        inputShape = [None, self.imgSize[1], self.imgSize[0], self.depth]   # Shape: [BatchSize, height, width, depth]
        self.placeholders = (tf1.placeholder( tf.float32, shape=inputShape, name='InputImages'),)
        return self.placeholders

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        assert inShape==None, "inShape must be None for input layers!"
        self.inShape = None
        self.outShape = [self.imgSize[1], self.imgSize[0], self.depth]
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        shortStr = 'Image Size: %dx%dx%d'%(self.imgSize[0], self.imgSize[1], self.depth)
        return shortStr
    
    # ******************************************************************************************************************
    def getInputStr(self):
        return '%s images of size %dx%d'%("Monochrome" if self.depth==1 else "Color", self.imgSize[0], self.imgSize[1])

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        # input is a tuple (of placeholders) with one tensor.
        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = [ input[0] ]
        return input[0], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        onnxInShape = (-1, self.depth, self.imgSize[1], self.imgSize[0]) # ONNX is channel-first
        doc = ("A list of %dx%d %s images. The images must be fed to the model as channel-first tensors. " +
               "(Shape: batchSize x %d x %d x %d)")%(self.imgSize[0], self.imgSize[1],
                                                     "monochrome" if self.depth==1 else "color",
                                                     self.depth, self.imgSize[1], self.imgSize[0])
        onnxBuilder.addParam('InputImage', 'float', onnxInShape, paramType='input', docStr=doc)
        return 'InputImage'
        
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        # Note 1: CoreML is channel-first.
        # Note 2: Batch Size is always 1. (Infer one image at a time)
        inputShape = (1, self.outShape[2], self.outShape[0], self.outShape[1])
        desc = ("A %dx%d %s image.")%(self.imgSize[0], self.imgSize[1],"monochrome" if self.depth==1 else "color")
        del cmlBuilder.spec.description.input[-1]   # Delete the dummy input
        cmlBuilder.addInput('InputImage', inputShape, 'float', desc)
        
        if self.depth==1:       # Grayscale
            # processed image = scale * image + bias
            cmlBuilder.set_pre_processing_parameters(image_input_names=["InputImage"],
                                                     gray_bias=cmlBuilder.rgbBias, image_scale=cmlBuilder.scale)
        else:                   # Color
            # processed image = scale * image + bias
            if type(cmlBuilder.rgbBias) == list:    rb, gb, bb = cmlBuilder.rgbBias
            else:                                   rb = gb = bb = cmlBuilder.rgbBias
            cmlBuilder.set_pre_processing_parameters(image_input_names=["InputImage"], is_bgr=cmlBuilder.isBgr,
                                                     red_bias=rb, green_bias=gb, blue_bias=bb,
                                                     image_scale=cmlBuilder.scale)
        return 'InputImage'

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        tfBuilder.addToInit("self.modelInput = tf1.placeholder(tf.float32, shape=[None, %d, %d, %d], "\
                            "name='InputImages')", (self.imgSize[1], self.imgSize[0], self.depth))
        
        tfBuilder.addToInfer("feedDic = { self.modelInput: samples }")
        tfBuilder.addToTrain("feedDic = { self.modelInput: batchSamples }")
        
# **********************************************************************************************************************
class TensorInLayer(Layer):
    argsDic = {
                's': ('shape', 'u*?', None),     # Example: Vector of size 10: use S10 => shape=[10],
              }                                  #          or 3x5 matrix: use S3/5 => shape = [3,5]
    orderedKeys = 's'
    name = 'TENSOR'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isInput = True

    # ******************************************************************************************************************
    def makePlaceholders(self):
        inputShape = [None] + self.shape
        self.placeholders = (tf1.placeholder( tf.float32, shape=inputShape, name='InputTensors'), )
        return self.placeholders
        
    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = self.outShape = self.shape
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        if len(self.shape) == 1:    shortStr = 'Vector Size: %d'%(self.shape[0])
        elif len(self.shape) == 2:  shortStr = 'Matrix Shape: ' + 'x'.join(str(x) for x in self.shape)
        else:                       shortStr = 'Tensor Shape: ' + 'x'.join(str(x) for x in self.shape)
        return shortStr

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        # input is a tuple (of placeholders) with one tensor.
        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = [ input[0] ]
        return input[0], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        if len(self.shape) == 1:
            inputName = 'InputVector'
            doc = "A vector of length %d."%(self.shape[0])
        elif len(self.shape) == 2:
            inputName = 'InputMatrix'
            doc = "A %d-row by %d-column matrix."%(self.shape[0], self.shape[1])
        else:
            inputName = 'InputTensor'
            doc = "A tensor of shape " + 'x'.join(str(x) for x in self.shape) + "."
            
        onnxBuilder.addParam(inputName, 'float', [-1]+self.shape, paramType='input', docStr=doc)
        return inputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        # Note: Batch Size is always 1. (Infer one image at a time)
        inputShape = (1,) + tuple(self.shape)
        del cmlBuilder.spec.description.input[-1]   # Delete the dummy input
        if len(self.shape) == 1:
            cmlBuilder.addInput('InputVector', inputShape, 'float', "A vector of length %d."%(self.shape[0]))
            return 'InputVector'
            
        if len(self.shape) == 2:
            cmlBuilder.addInput('InputMatrix', inputShape, 'float',
                                "A %d-row by %d-column matrix."%(self.shape[0], self.shape[1]))
            return 'InputMatrix'
        
        cmlBuilder.addInput('InputTensor', inputShape, 'float',
                            "A tensor of shape " + 'x'.join(str(x) for x in self.shape) + ".")
        return 'InputTensor'

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if len(self.shape) == 1:    inputName = 'InputVector'
        elif len(self.shape) == 2:  inputName = 'InputMatrix'
        else:                       inputName = 'InputTensor'
        tfBuilder.addToInit("self.modelInput = tf1.placeholder( tf.float32, shape=%s, name='%s')",
                            (str([None] + self.shape), inputName))
                            
        tfBuilder.addToInfer("feedDic = { self.modelInput: samples }")
        tfBuilder.addToTrain("feedDic = { self.modelInput: batchSamples }")

# **********************************************************************************************************************
class EmbeddingInLayer(Layer):
    argsDic = {
                'o': ('outSize', 'u', None),
                's': ('initStd', 'f', 0.02),    # Standard Dev. for initializers
                'l': ('maxLen', 'u', 512),      # Max Sequence Length (Default: 512)
                'v': ('vocabSize', 'u', 30522), # Vocab Size (Default: 30522)
                'r': ('rank', 'u', 0),          # Used only for word embedding (The big matrix)
              }
    orderedKeys = 'toslvr'
    name = 'EMB'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isInput = True
        # IMPORTANT NOTE:
        # There are different types of sequence length:
        # Model's maxSeqLen:        This is fixed for the a model design and used during the training of the model.
        #                           The maximum sequence that can be handled by the model. For example for BERTbase,
        #                           this is set to 512.
        # Datasets's maxSeqLen:     This is the max sequence length that occurs in a dataset. For example for SQuAD,
        #                           this is set to 384. Must be less than the Model's maxSeqLen (The "maxLen" argument
        #                           of this layer)
        # seqLen:                   This is the sequence length for a single sample processed by the model. It may
        #                           or may not include padding. For processing just one sample, there is not need for
        #                           padding. To process a batch of samples, we use padding to make them the same
        #                           length.
        # noPadLen:                 When padding is used, this is the non-padded sequence length. When padding is not
        #                           used, this is equal to the sequence length. (When processing only one sample for
        #                           example)

    # ******************************************************************************************************************
    def makePlaceholders(self):
        self.placeholders = None
        self.placeholders = (
            tf1.placeholder( tf.int32, shape=[None, None], name='TokenIds'),        # Shape: [BatchSize, SeqLen]
            tf1.placeholder( tf.int32, shape=[None, None], name='TokenTypes'))      # Shape: [BatchSize, SeqLen]
                
        return self.placeholders
            
    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        assert inShape==None, "inShape must be None for input layers!"
        self.inShape = [ -1, 2 ]    # -1 means variable sequence Len, 2 is for TokenIds, TokenTypes
        self.outShape = [ -1, self.outSize ]
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        if self.rank>0: return 'LR%d'%(self.rank)
        return ''
    
    # ******************************************************************************************************************
    def getInputStr(self):
        return 'A tuple of TokenIds and TokenTypes.'
        
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        paramStrs = None
        if includeSizeInfo:
            if self.rank==0:
                paramStrs = ['%s/W %dx%d'%(self.scope, self.vocabSize, self.outSize),       # Words Lookup table
                             '%s/T %dx%d'%(self.scope, 2, self.outSize),                    # Types Lookup table
                             '%s/P %dx%d'%(self.scope, self.maxLen, self.outSize)]          # Positions Lookup table
            else:
                paramStrs = ['%s/G %dx%d'%(self.scope, self.vocabSize, self.rank),          # Words Lookup table (G)
                             '%s/H %dx%d'%(self.scope, self.rank, self.outSize),            # Words Lookup table (H)
                             '%s/T %dx%d'%(self.scope, 2, self.outSize),                    # Types Lookup table
                             '%s/P %dx%d'%(self.scope, self.maxLen, self.outSize)]          # Positions Lookup table
                
        elif self.rank==0:
            paramStrs = [self.scope + '/W', self.scope + '/T', self.scope + '/P']
        else:
            paramStrs = [self.scope + '/G', self.scope + '/H', self.scope + '/T', self.scope + '/P']

            
        return paramStrs

    # ******************************************************************************************************************
    def inferRank(self, initValues):
        if self.rank > 0:                                       return  # We already know the rank
        if initValues is None:                                  return  # Can only infer rank from initValues
        
        shape0 = self.vocabSize
        assert initValues[0].shape[0] == shape0,                    \
            "%s: VocabSize mismatch! (%d vs %d)"%(self.scope, initValues[0].shape[0], shape0)
        if initValues[0].shape[1] == self.outSize:                  return  # Not Decomposed
        assert initValues[0].shape[1] == initValues[1].shape[0],    \
            "%s: LR Init tensor rank mismatch. (%d vs %d)!"%(self.scope, initValues[0].shape[1], initValues[1].shape[0])
        assert initValues[1].shape[1] == self.outSize,              \
            "%s: LR Init tensor outSize mismatch! (%d vs %d)"%(self.scope, initValues[1].shape[1], self.outSize)
        # Now we know it is decomposed => Set rank
        self.rank = initValues[1].shape[0]

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.inferRank(initValues)
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            if initValues is None:
                if self.rank == 0:
                    initW = tfr.truncated_normal([self.vocabSize, self.outSize], mean=0, stddev=self.initStd, seed=SEED)
                else:
                    initG = tfr.truncated_normal([self.vocabSize, self.rank], mean=0, stddev=self.initStd, seed=SEED)
                    initH = tfr.truncated_normal([self.rank, self.outSize], mean=0, stddev=self.initStd, seed=SEED)
                initT = tfr.truncated_normal([2, self.outSize], mean=0, stddev=self.initStd, seed=SEED)
                initP = tfr.truncated_normal([self.maxLen, self.outSize], mean=0, stddev=self.initStd, seed=SEED)
            else:
                if self.rank == 0:
                    initW, initT, initP = initValues[0:3]
                    self.checkShape('W', initW.shape, (self.vocabSize, self.outSize))
                else:
                    initG, initH, initT, initP = initValues[0:4]
                    self.checkShape('G', initG.shape, (self.vocabSize, self.rank) )
                    self.checkShape('H', initH.shape, (self.rank, self.outSize))
                self.checkShape('T', initT.shape, (2, self.outSize))
                self.checkShape('P', initP.shape, (self.maxLen, self.outSize) )
                
            self.netParams = []
            if self.rank == 0:
                self.wordTable = self.makeNetParam('W', initW)
            else:
                self.wordTable = self.makeNetParam('G', initG)
                self.wordTableH = self.makeNetParam('H', initH)
                
            self.typeTable = self.makeNetParam('T', initT)
            self.posTable = self.makeNetParam('P', initP)
                                                            
        return self.netParams

    # ******************************************************************************************************************
    def embed(self, sigFata):
        return self.model.session.run(self.embeddings, feed_dict={self.placeholders[0]: sigFata})
    
    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        with tf.name_scope(self.scope):
            tokenIds, tokenTypes = input
            noPadLens = tf.reduce_sum(tf.minimum(tokenIds, 1),axis=1)
            
            tokShape = tf.shape(tokenIds)
            batchSize, seqLen = tokShape[0], tokShape[1]

            # Create Attention Masks:
            def getAttMask2D(noPadSeqLen):
                # This returns 2D (seqLen x seqLen) matrix for one of the BatchSize sequences. It has 0 where we want to
                # keep attention and -10000 where we want to mask attention. The output is seqLen x SeqLen.
                # To create this, we are padding a square noPadLens x noPadLens tensor of zeros with values of -10000 at
                # right and bottom.
                # If there is no padding, it will be all zeros (noPadSeqLens=seqLen => numMasked=0)
                numMasked = seqLen-noPadSeqLen
                return tf.pad(tf.zeros([noPadSeqLen,noPadSeqLen], tf.float32),
                              [[0,numMasked],[0,numMasked]], "CONSTANT", constant_values=-10000.0)  # Shape: [seqLen, SeqLen]
            attMasks3D = tf.map_fn(getAttMask2D, noPadLens, dtype=tf.float32)    # Shape: [BatchSize, seqLen, SeqLen
        
            # Now reshape the attention masks so that it is ready to be added to the actual attentions. (See
            # BertLayer::buildGraph)
            self.layers.attentionMasks = tf.reshape(attMasks3D, [-1, 1, seqLen, seqLen])  # Shape: [BatchSize, 1, seqLen, SeqLen]

            # 1: Embedding the tokens:
            flatIds = tf.reshape(tokenIds, [-1])
            wordEmb = tf.gather(self.wordTable, flatIds)
            if self.rank>0:  wordEmb = tf.matmul(wordEmb, self.wordTableH)
            outEmb = tf.reshape(wordEmb, [batchSize, seqLen, self.outSize])
                        
            # 2: Embedding token types. The vocab is small (only 0 and 1), so we always do
            # one-hot here, since it is always faster for a small vocabulary.
            flatTypeIds = tf.reshape(tokenTypes, [-1])
            oneHotTypeIds = tf.one_hot(flatTypeIds, depth=2)
            typeEmb = tf.matmul(oneHotTypeIds, self.typeTable)
            outEmb += tf.reshape(typeEmb, [batchSize, seqLen, self.outSize])

            # 3: Embedding Positions
            # The table is learned with max position size. Since the indexes here are: [0, 1, 2, ... seqLen-1],
            # looking up the table is the same as slicing it which is faster.
            posEmb = tf.slice(self.posTable, [0, 0], [seqLen, -1])      # Shape: [seqLen, outSize]
            
            # To broadcast this to all items in the batch we need to reshape it:
            outEmb += tf.reshape(posEmb, [1, seqLen, self.outSize])

            outputs = [ self.layers.attentionMasks, outEmb ]
            self.buildPostActivation(outputs, isTraining)

            outKey = 'training' if isTraining else 'inference'
            self.layerOuts[outKey] = outputs
            return outputs[-1], None
            
    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        # Get seqLen from onnxBuilder
        rs = self.deepScope()   # RootScope
        wName, wNameH, tName, pName = ((rs+x) for x in ['WordTable', 'WordTableH', 'TypeTable', 'PosTable'])
        params = self.npNetParams

        doc = ("A list of token IDs with integer values between 0 and %d. If the inputs are fed to the model in " +
               "batches of more than one, they must be padded so that the lists of token IDs for each sample in the " +
               "batch has the same length. In either case the length of token ID lists must never " +
               "exceed %d.")%(self.vocabSize-1, self.maxLen)
        onnxBuilder.addParam('TokIds', 'int32', [-1,-1], paramType='input', docStr=doc)
        doc = ("A list of token types with integer values 0 or 1. If the inputs are fed to the model in batches " +
               "of more than one, they must be padded so that the lists of token types for each sample in the " +
               "batch has the same length. This must always be the same length as 'TokIds' input.")
        onnxBuilder.addParam('TokTypes', 'int32', [-1,-1], paramType='input', docStr=doc)

        onnxBuilder.addParam(rs+'OtherParams', 'int64', [4], [self.vocabSize,self.outSize,1,-1])
        onnxBuilder.addNode('Shape', ['TokIds'], [rs+'inShape'], rs+'Shape')
        onnxBuilder.addNode('Concat', [rs+'inShape', rs+'OtherParams'], ['Dimensions'], rs+'Concat', axis=0)
       
        onnxBuilder.makeShape(rs+'OutShape', 'batchSize,seqLen,outSize')

        # Now build the attention mask: (barchSize x seqLen x seqLen)
        s = self.deepScope('AttMask')
        onnxBuilder.makeShape(s+'MaskShape', 'seqLen,batchSize,seqLen')
        onnxBuilder.addParam(s+'ClipTo1Val', 'float', [], [1.0])
        onnxBuilder.addParam(s+'MaskVal', 'float', [], [10000.0])

        onnxBuilder.addNode('Cast', ['TokIds'], [s+'tokIdFloats'], s+'Cast', to=1)  # 1 = TensorProto.FLOAT
        onnxBuilder.addNode('Min', [s+'tokIdFloats', s+'ClipTo1Val'], [s+'Clipped'], s+'Min')
        onnxBuilder.addNode('Expand', [s+'Clipped', s+'MaskShape'], [s+'Clipped3D0'], s+'Expand')
        onnxBuilder.addNode('Transpose', [s+'Clipped3D0'], [s+'Clipped3D'], s+'Transpose1', perm=[1,0,2])
        onnxBuilder.addNode('Transpose', [s+'Clipped3D'], [s+'Clipped3DT'], s+'Transpose2', perm=[0,2,1])
        onnxBuilder.addNode('Mul', [s+'Clipped3D',s+'Clipped3DT'], [s+'MaskOnes'], s+'Mul1')
        onnxBuilder.addNode('Mul', [s+'MaskOnes',s+'MaskVal'], [s+'InvertedMask'], s+'Mul2')
        onnxBuilder.addNode('Sub', [s+'InvertedMask',s+'MaskVal'], [s+'AttMask3D'], s+'Sub')
        onnxBuilder.addReshape(s+'AttMask3D', 'batchSize,1,seqLen,seqLen', 'AttMask')

        # Embedding for tokIds:
        s = self.deepScope('EmbeddingTokIds')
        onnxBuilder.addNetParam(wName, params[0])
        onnxBuilder.addNode('Squeeze', ['TokIds'], [s+'tokIdsFlat'], s+'Squeeze')
        if self.rank==0:
            onnxBuilder.addNode('Gather', [wName, s+'tokIdsFlat'], [s+'tokIdsEmb1D'], s+'Gather')
        else:
            onnxBuilder.addNetParam(wNameH, params[1])
            onnxBuilder.addNode('Gather', [wName, s+'tokIdsFlat'], [s+'tokIdsEmbG'], s+'Gather')
            onnxBuilder.addNode('MatMul', [s+'tokIdsEmbG', wNameH], [s+'tokIdsEmb1D'], s+'MatMul')
        onnxBuilder.addNode('Reshape', [s+'tokIdsEmb1D', rs+'OutShape'], [rs+'tokIdsEmb'], s+'Reshape')

        # Embedding for tokTypes:
        s = self.deepScope('EmbeddingTokTypes')
        onnxBuilder.addNetParam(tName, params[-2])
        onnxBuilder.addNode('Squeeze', ['TokTypes'], [s+'tokTypesFlat'], s+'Squeeze')
        onnxBuilder.addNode('Gather', [tName, s+'tokTypesFlat'], [s+'tokTypesEmb1D'], s+'Gather')
        onnxBuilder.addNode('Reshape', [s+'tokTypesEmb1D', rs+'OutShape'], [rs+'tokTypesEmb'], s+'Reshape')

        # Embedding for positions:
        s = self.deepScope('EmbeddingPositions')
        onnxBuilder.addNetParam(pName, params[-1])
        onnxBuilder.addParam(s+'SliceStarts', 'int64', [2], [0,0])
        onnxBuilder.makeShape(s+'SliceEnds', 'seqLen,outSize')
        onnxBuilder.addNode('Slice', [pName, s+'SliceStarts', s+'SliceEnds'], [s+'PosEmb1D'], s+'Slice')
        onnxBuilder.addReshape(s+'PosEmb1D', '1,seqLen,outSize', rs+'PosEmb')

        # Add all embeddings:
        onnxBuilder.addNode('Add', [rs+'tokIdsEmb', rs+'tokTypesEmb'], [rs+'Embeddings1'], rs+'Add1')
        onnxBuilder.addNode('Add', [rs+'Embeddings1', rs+'PosEmb'], [rs+'Embeddings'], rs+'Add2')

        return rs+'Embeddings'

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        # Note:
        #    Since this is an input layer, we ignore "inputName" and use the hardcoded names "TokIds" and "TokTypes"
        #    Also all the following BERT layers, use the hardcoded name 'Attention_Mask'
        
        del cmlBuilder.spec.description.input[-1]   # Delete the dummy input

        # We need seqLen. Get it from the builder:
        seqLen = cmlBuilder.maxSeqLen
        cmlBuilder.addInput("TokIds", (seqLen,), 'int32',
                            "Token IDs containing integer values between 0 and %d. The length of token IDs "\
                            "must be exactly %d."%(self.vocabSize-1, seqLen))
                             
        cmlBuilder.addInput("TokTypes", (seqLen,), 'int32',
                            "Token types containing integer values 0 or 1. This must always be the same "\
                            "length as 'TokIds' input (%d)."%(seqLen))

        # First reshape TokIds and TokTypes to 5D tensors
        cmlBuilder.add_reshape_static('TokIds5D', "TokIds",
                               'TokIds5D', (cmlBuilder.maxSeqLen,1,1,1,1))              # Shape: seqLen,1,1,1,1
        cmlBuilder.add_reshape_static('TokTypes5D', "TokTypes",
                               'TokTypes5D', (cmlBuilder.maxSeqLen,1,1,1,1))            # Shape: seqLen,1,1,1,1

        # First Build Attention Mask:
        cmlBuilder.add_clip('Attention_TokIds01', "TokIds5D", 'Attention_TokIds01', max_value=1) # Shape: 1,seqLen,1,1
        
        cmlBuilder.add_reshape('Attention_NoPad1D', "Attention_TokIds01",
                               'Attention_NoPad1D', (1,seqLen,1,1), mode=0)                 # Shape: 1,seqLen,1,1
        
        cmlBuilder.add_broadcast_to_static('Attention_NoPad2D', "Attention_NoPad1D",
                                           'Attention_NoPad2D', (1,seqLen,seqLen,1))        # Shape: 1,seqLen,seqLen,1
        
        cmlBuilder.add_transpose('Attention_NoPad2DT', (0,2,1,3), "Attention_NoPad2D",      # Shape: 1,seqLen,seqLen,1
                                 'Attention_NoPad2DT')

        cmlBuilder.add_elementwise('Attention_NoPad', ["Attention_NoPad2D", "Attention_NoPad2DT"],
                                   'Attention_NoPad', 'MULTIPLY')
                                
        cmlBuilder.add_scale('Attention_Mask0', 10000, -10000, True, "Attention_NoPad",
                             'Attention_Mask0', shape_scale=[1], shape_bias=[1])
        cmlBuilder.add_squeeze('Attention_Mask', "Attention_Mask0",
                               'Attention_Mask', squeeze_all=True)                          # Shape: seqLen,seqLen
 
        # Embedding for TokIds:
        if self.rank==0:
            wTok, wType, wPos = NetParam.toNp(self.netParams, self.model.session)
            TokIdEmbName = self.scope+'_TokIdEmb'
            cmlBuilder.add_embedding(name=TokIdEmbName, W=wTok.getCoreMlWeight('T'),        # Shape: seqLen,1,outSize,1,1
                                    b=None, has_bias=False,
                                    input_dim=self.vocabSize, output_channels=self.outSize,
                                    input_name="TokIds5D", output_name=TokIdEmbName, **wTok.getCoreMlQuantInfo())
        else:
            gTok, hTok, wType, wPos = NetParam.toNp(self.netParams, self.model.session)
            outputName = self.scope+'_TokIdEmbG'
            cmlBuilder.add_embedding(name=outputName, W=gTok.getCoreMlWeight('T'),          # Shape: seqLen,1,rank,1,1
                                     b=None, has_bias=False,
                                     input_dim=self.vocabSize, output_channels=self.rank,
                                     input_name="TokIds5D", output_name=outputName, **gTok.getCoreMlQuantInfo())
            inputName, TokIdEmbName = outputName, self.scope+'_TokIdEmb'
            cmlBuilder.add_inner_product(name=TokIdEmbName, W=hTok.getCoreMlWeight('T'),    # Shape: seqLen,1,outSize,1,1
                                         b=None, has_bias=False,
                                         input_channels=self.rank, output_channels=self.outSize,
                                         input_name=inputName, output_name=TokIdEmbName, **hTok.getCoreMlQuantInfo())

        # Embedding for TokTypes:
        tokTypeEmbName = self.scope+'_TokTypeEmb'
        cmlBuilder.add_embedding(name=tokTypeEmbName, W=wType.getCoreMlWeight('T'),         # Shape: seqLen,1,outSize,1,1
                                 b=None, has_bias=False,
                                 input_dim=2, output_channels=self.outSize,
                                 input_name="TokTypes5D", output_name=tokTypeEmbName, **wType.getCoreMlQuantInfo())

        # Embedding for Positions:
        outputName = self.scope+'_PosEmbW'
        cmlBuilder.add_load_constant(outputName, outputName,
                                     wPos.value()[:seqLen], (seqLen, self.outSize, 1))
        
        inputName, posEmbName = outputName, self.scope+'_PosEmb'
        cmlBuilder.add_reshape(posEmbName, inputName, posEmbName,
                               (seqLen, self.outSize, 1, 1), mode=0)                    # Shape: seqLen,1,outSize,1,1

        outputName = self.scope
        cmlBuilder.add_elementwise(outputName, [TokIdEmbName, tokTypeEmbName, posEmbName],
                                   outputName, 'ADD')                                   # Shape: seqLen,1,outSize,1,1

        return outputName
        
    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        tfBuilder.addToInit(("self.attentionMasks = None",
                             "self.modelInput = (tf1.placeholder( tf.int32, shape=[None, None], name='TokenIds'),",
                             "                   tf1.placeholder( tf.int32, shape=[None, None], name='TokenTypes'))"))

        tfBuilder.addToInfer(("# 'samples' must be a tuple of 2 numpy arrays for 'TokenIds' and 'TokenTypes'",
                              "feedDic = { self.modelInput[0]: samples[0], self.modelInput[1]: samples[1] }",
                              ""))
                              
        tfBuilder.addToTrain(("# 'batchSamples' must be a tuple of 2 numpy arrays for 'TokenIds' and 'TokenTypes'",
                              "feedDic = { self.modelInput[0]: batchSamples[0], self.modelInput[1]: batchSamples[1] }",
                              ""))
                              
        tfBuilder.addToGraph(("tokenIds, tokenTypes = self.modelInput",
                              "noPadLens = tf.reduce_sum(tf.minimum(tokenIds, 1),axis=1)",
                              tfBuilder.getScopeStr(self.scope)))
        tfBuilder.graphIndent += 1
        
        if self.rank == 0:
            tfBuilder.addToGraph("wordTable = self.makeVariable('W', [%d, %d], %s, %d)",
                                 (self.vocabSize, self.outSize, str(self.initStd),
                                  self.netParams[0].codebookSize if tfBuilder.runQuantized else 0))
        else:
            tfBuilder.addToGraph(("wordTable = self.makeVariable('G', [%d, %d], %s, %d)",
                                  "wordTableH = self.makeVariable('H', [%d, %d], %s, %d)"),
                                 (self.vocabSize, self.rank, str(self.initStd),
                                  self.netParams[0].codebookSize if tfBuilder.runQuantized else 0,
                                  self.rank, self.outSize, str(self.initStd),
                                  self.netParams[1].codebookSize if tfBuilder.runQuantized else 0))

        tfBuilder.addToGraph(("typeTable = self.makeVariable('T', [2, %d], %s, %d)",
                              "posTable = self.makeVariable('P', [%d, %d], %s, %d)"),
                             (self.outSize, str(self.initStd),
                              self.netParams[-2].codebookSize if tfBuilder.runQuantized else 0,
                              self.maxLen, self.outSize, str(self.initStd),
                              self.netParams[-1].codebookSize if tfBuilder.runQuantized else 0))

        tfBuilder.addToGraph(("tokShape = tf.shape(tokenIds)",
                              "batchSize, seqLen = tokShape[0], tokShape[1]",
                              "",
                              "def getAttMask2D(noPadSeqLen):",
                              "    numMasked = seqLen-noPadSeqLen",
                              "    return tf.pad(tf.zeros([noPadSeqLen,noPadSeqLen], tf.float32),",
                              "                  [[0,numMasked],[0,numMasked]],",
                              "                  \"CONSTANT\",",
                              "                  constant_values=-10000.0)",
                              "",
                              "attMasks3D = tf.map_fn(getAttMask2D, noPadLens, dtype=tf.float32)",
                              "self.attentionMasks = tf.reshape(attMasks3D, [-1, 1, seqLen, seqLen])",
                              "",
                              "# 1: Embedding the tokens:",
                              "flatIds = tf.reshape(tokenIds, [-1])",
                              "wordEmb = tf.gather(wordTable, flatIds)",
                              "" if self.rank==0 else "wordEmb = tf.matmul(wordEmb, wordTableH)",
                              "out = tf.reshape(wordEmb, [batchSize, seqLen, %d])"%(self.outSize),
                              "",
                              "# 2: Embedding token types.",
                              "flatTypeIds = tf.reshape(tokenTypes, [-1])",
                              "oneHotTypeIds = tf.one_hot(flatTypeIds, depth=2)",
                              "typeEmb = tf.matmul(oneHotTypeIds, typeTable)",
                              "out += tf.reshape(typeEmb, [batchSize, seqLen, %d])"%(self.outSize),
                              "",
                              "# 3: Embedding Positions",
                              "posEmb = tf.slice(posTable, [0, 0], [seqLen, -1])",
                              "out += tf.reshape(posEmb, [1, seqLen, %d])"%(self.outSize),
                              ""))
        tfBuilder.graphIndent -= 1
            
    
    # ******************************************************************************************************************
    def decompose(self, session, decInfo, decomposeDWCN=True):
        # decInfo:  decType, rank, moreInfo
        #           decType:    'lr' or 'ldr'
        #           rank:       int->rank or float->MSE Upper Bound (Find best rank)
        #           moreInfo:   (e,f) (ignored for LR)
        assert (self.isDecomposed() == False), "%s: Layer already decomposed!"%(self.scope)
        
        decType, rank, decMode = decInfo
        w, t, p = NetParam.toNpValues(self.netParams, session)
        
        assert decType == 'lr', "%s: Only Low-Rank Decomposition is supported for '%s' layers."%(self.scope, self.name)
        
        # NOTE: We only decompose the Word Embedding. Leave Type and Position embedding unchanged.
        rank, g, h, mse = self.decomposeMatrix(w, rank)
        if (g is None) or (h is None):
            if mse is None:
                maxRank = int(np.prod(w.shape)/sum(w.shape))
                return (None, None, None,
                    '%s => Cannot Decompose, Rank(%d)>MaxRank(%d)'%(self.scope, rank, maxRank))
                    
            mseUB, mseHi, rankHi = float(rank), mse, rank
            return (None, None, None,
                    '%s => Cannot Decompose, MSE(%d)=%f>%f'%(self.scope, rankHi, mseHi, mseUB))

        layerStrParts = self.getLayerStr().split(':')
        layerStrParts[0] = layerStrParts[0] + '_R%d'%(rank)
        newLayerStr = ':'.join(layerStrParts)
        
        numOldParams = np.prod(w.shape)
        numNewParams = rank*sum(w.shape)
        assert (numOldParams>numNewParams)  # Just to be sure. This should never happen.
        
        changeStr = 'Reduction: %.1f%%'%((numOldParams-numNewParams)*100.0/numOldParams)
        infoStr = '%s => LR(%d), MSE=%f, Params: %d->%d (%s)'%(self.scope, rank, mse, numOldParams, numNewParams, changeStr)
        
        return ([g, h, t, p], newLayerStr, numNewParams + t.size + p.size, infoStr)

# **********************************************************************************************************************
# MARK: ------------------------ Hidden Layers ------------------------
class FcLayer(Layer):
    argsDic = {
                'o': ('outSize', 'u', None),
                'r': ('rank', 'u', 0),              # Rank=0 => Not Decomposed (Default)
                'b': ('hasBias', 'u', 1),
                'l': ('decType', 'dec', 'lr'),      # Current Decomposition types: lr, ldr
                'e': ('e', 'f', -2.0),
                'f': ('f', 'f', 2.0),
              }
    orderedKeys = 'orblef'
    name = 'FC'
    # ******************************************************************************************************************
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.flatInputSize = None
        
    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]
        self.outShape = [ self.outSize ]
        self.flatInputSize = np.abs(np.prod(self.inShape))
        
        if len(self.inShape)==2:
            # The shape of BERT output is [seqLen, outSize]. seqLen can be -1 which means a variable
            # sequence length. The outShape of FC layer in the case remains 2-D until a Transformer Pooling
            # layer is used to flatten it. After that the outShape becomes 1D like normal FC layers.
            self.flatInputSize = self.inShape[1]
            self.outShape = [ -1, self.outSize ]

        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        shortStr = ''
        if self.rank > 0:
            if self.decType == 'lr':    shortStr += 'LR%d'%(self.rank)
            elif self.decType == 'ldr': shortStr += 'LDR%d e:%d f:%d'%(self.rank, self.e, self.f)

        if self.hasBias==0:             shortStr += 'No Biases'
        return shortStr
    
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        if self.rank > 0:
            if includeSizeInfo:
                paramStrs = ['%s/G %dx%d'%(self.scope, self.flatInputSize, self.rank),
                             '%s/H %dx%d'%(self.scope, self.rank, self.outSize)]
            else:
                paramStrs = [self.scope + '/G', self.scope + '/H']
        elif includeSizeInfo:
            paramStrs = [ '%s/Weights: %dx%d'%(self.scope, self.flatInputSize, self.outSize) ]
        else:
            paramStrs = [ self.scope + '/Weights' ]

        if self.hasBias:
            paramStrs += ['%s/Biases%s'%(self.scope, (' %d'%(self.outSize) if includeSizeInfo else ''))]
            
        return paramStrs

    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        listOfL2Losses = [ netParam.tfL2Loss() for netParam in self.netParams ]
        self.l2Loss = factor * tf.add_n(listOfL2Losses)
    
    # ******************************************************************************************************************
    def inferRank(self, initValues):
        if self.rank > 0:                                       return  # It is LR/LDR and we already know the rank
        if initValues is None:                                  return  # No Init Values, Nothing to infer from
        if initValues[0].shape[0] == self.flatInputSize:
            if initValues[0].shape[1] == self.outSize:          return  # Not an LR/LDR layer
            # Must be an LR layer:
            assert initValues[0].shape[1] == initValues[1].shape[0],    \
                "%s: LR Init tensor rank mismatch. (%d vs %d)!"%(self.scope, initValues[0].shape[1], initValues[1].shape[0])
            assert initValues[1].shape[1] == self.outSize,              \
                "%s: LR Init tensor outSize mismatch! (%d vs %d)"%(self.scope, initValues[1].shape[1], self.outSize)
            self.rank = initValues[1].shape[0]
            return

        # Must be an LDR layer:
        assert initValues[0].shape[1] == self.outSize, "%s: Invalid init value shapes for LDR layer!"%(self.scope)
        assert initValues[0].shape[0] == initValues[1].shape[0], "%s: Invalid init value shapes for LDR layer!"%(self.scope)
        assert initValues[1].shape[1] == self.flatInputSize, "%s: Invalid init value shapes for LDR layer!"%(self.scope)
        self.rank = initValues[1].shape[0]
        self.decType == 'ldr'

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.inferRank(initValues)
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            if initValues is None:
                if self.rank == 0:
                    initW = tfr.truncated_normal([self.flatInputSize, self.outSize], mean=0, stddev=1.0/np.sqrt(self.flatInputSize), seed=SEED)
                elif self.decType == 'lr':
                    initG = tfr.truncated_normal([self.flatInputSize, self.rank], mean=0, stddev=1.0/np.sqrt(self.flatInputSize), seed=SEED)
                    initH = tfr.truncated_normal([self.rank, self.outSize], mean=0, stddev=1.0/np.sqrt(self.flatInputSize), seed=SEED)
                elif self.decType == 'ldr':
                    initG = tfr.truncated_normal([self.rank, self.outSize], mean=0, stddev=1.0/np.sqrt(self.flatInputSize), seed=SEED)
                    initH = tfr.truncated_normal([self.rank, self.flatInputSize], mean=0, stddev=1.0/np.sqrt(self.rank), seed=SEED)
                if self.hasBias:    initB = tf.constant(0.0, shape=[self.outSize])
            else:
                if self.rank > 0:
                    initG, initH = initValues[0], initValues[1]
                    if self.decType == 'lr':
                        self.checkShape('G', initG.shape, (self.flatInputSize, self.rank) )
                        self.checkShape('H', initH.shape, (self.rank, self.outSize))
                    else:
                        self.checkShape('G', initG.shape, (self.rank, self.outSize))
                        self.checkShape('H', initH.shape, (self.rank, self.flatInputSize))
                else:
                    initW = initValues[0]
                    self.checkShape('Weights', initW.shape, (self.flatInputSize, self.outSize))
                if self.hasBias:
                    initB = initValues[2] if self.rank>0 else initValues[1]
                    self.checkShape('Biases', initB.shape, (self.outSize,))

            self.netParams = []
            if self.rank > 0:
                self.g = self.makeNetParam('G', initG)
                self.h = self.makeNetParam('H', initH)
            else:
                self.weights = self.makeNetParam('Weights', initW)

            if self.hasBias:
                self.biases = self.makeNetParam('Biases', initB)
        
        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        outputs = []
        with tf.name_scope(self.scope):
            # We may need to flatten the input:
            flattenedInput = tf.reshape(input, [-1, self.flatInputSize], name='Flatten')
            if self.rank==0:
                wx = tf.matmul(flattenedInput, self.weights, name='xW')
            elif self.decType == 'lr':
                wx = tf.matmul(flattenedInput, self.g, name='xG')
                wx = tf.matmul(wx, self.h, name='xGH')
            elif self.decType == 'ldr':
                wx = ldr.getLdrGraph(True, flattenedInput, self.g, self.h, self.e, self.f, self.outSize, self.flatInputSize, self.rank)
            
            if self.hasBias:    outputs += [ tf.add(wx, self.biases, name='xWplusB' if self.rank==0 else 'xGHplusB') ]
            else:               outputs += [ wx ]

            if len(self.inShape)==2:
                # For NLP models, we reshape the output to [BatchSize, seqLen, outSize]
                seqLen = tf.shape(input)[1]
                outputs[-1] = tf.reshape(outputs[-1], [-1, seqLen, self.outSize])

            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        if self.isVectorIn() == False:   # Need to flatten the input
            inputName = onnxBuilder.addReshape(inputName, [-1,self.flatInputSize], s + 'FlattenedInput')

        params = self.npNetParams
        param0 = params[0]
        if len(self.inShape)==3:
            # If this is the first dense layer after Convolutional layers, the flattening above requires
            # some special re-ordering of the weight (or G) matrix.
            tempShape = self.inShape + [param0.shape[1]]
            rawVal = np.transpose( param0.rawVal.reshape(tempShape),(2,0,1,3) ).reshape(param0.shape)
            bitMask = None
            if param0.bitMask is not None:
                bitMask = np.transpose( param0.bitMask.reshape(tempShape),(2,0,1,3) ).reshape(param0.shape)
            param0 = NetParam('NP', rawVal, param0.codebook, bitMask, param0.trainable, param0.name)
            
        wName, gName, hName, bName = ((s+x) for x in ['Weights', 'G', 'H', 'Biases'])
        bNameList = [bName] if self.hasBias else []
        
        if self.rank > 0:
            onnxBuilder.addNetParam(gName, param0)
            onnxBuilder.addNetParam(hName, params[1])

            outputName = s + "xG"
            onnxBuilder.addNode('MatMul', [inputName, gName], [outputName], s+'MatMulG')
            inputName = outputName

            # Using the default values for the attributes: alpha=1, beta=1, transA=0, transB=0
            outputName = s + ("xGHplusB" if self.hasBias else "xGH")
            onnxBuilder.addNode('Gemm', [inputName, hName] + bNameList, [outputName], s+'GemmH')
        else:
            onnxBuilder.addNetParam(wName, param0)
            outputName = s + ("xWplusB" if self.hasBias else "xW")
            onnxBuilder.addNode('Gemm', [inputName, wName] + bNameList, [outputName], s+'Gemm')

        if self.hasBias:
            onnxBuilder.addNetParam(bName, params[-1])

        if len(self.inShape)==2:
            # For NLP models, we reshape the output to [BatchSize, seqLen, outSize]
            outputName = onnxBuilder.addReshape(outputName, 'batchSize,seqLen,-1')

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName
        
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        if self.isVectorIn() == False:   # Need to flatten the input
            outputName = inputName + "Flat"
            cmlBuilder.add_flatten(s+'flatten', 1, inputName, outputName)
            inputName = outputName
            
        params = NetParam.toNp(self.netParams, self.model.session)
        if self.rank > 0:
            g, h, b = params if self.hasBias else (params[0], params[1], None)
            outputName = s + "xG"
            cmlBuilder.add_inner_product(name =            s+'inner_productG',
                                         W =               g.getCoreMlWeight('T'),
                                         b =               None,
                                         input_channels =  self.flatInputSize,
                                         output_channels = self.rank,
                                         has_bias =        False,
                                         input_name =      inputName,
                                         output_name =     outputName,
                                         **g.getCoreMlQuantInfo())
            inputName = outputName
            outputName = s + ("xGHplusB" if self.hasBias else "xGH")
            cmlBuilder.add_inner_product(name =            s+'inner_productH',
                                         W =               h.getCoreMlWeight('T'),
                                         b =               None if b is None else b.value(),
                                         input_channels =  self.rank,
                                         output_channels = self.outSize,
                                         has_bias =        self.hasBias,
                                         input_name =      inputName,
                                         output_name =     outputName,
                                         **h.getCoreMlQuantInfo())
        else:
            w, b = params if self.hasBias else (params[0], None)
            outputName = s + ("xWplusB" if self.hasBias else "xW")
            cmlBuilder.add_inner_product(name =            s+'inner_product',
                                         W =               w.getCoreMlWeight('T'),
                                         b =               None if b is None else b.value(),
                                         input_channels =  self.flatInputSize,
                                         output_channels = self.outSize,
                                         has_bias =        self.hasBias,
                                         input_name =      inputName,
                                         output_name =     outputName,
                                         **w.getCoreMlQuantInfo())

        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('fcLayer'):
            tfBuilder.defineMethod('fcLayer')
            tfBuilder.addMethod(("def fcLayer(self, layerIn, shape, hasBias, l2LossFactor, cb=[0,0]):",
                                 "    flattenedInput = tf.reshape(layerIn, [-1, shape[0]], name='Flatten')",
                                 "    if len(shape)==3:",
                                 "        g = self.makeVariable('G', shape[:2], 1.0/np.sqrt(shape[0]), cb[0])",
                                 "        h = self.makeVariable('H', shape[1:], 1.0/np.sqrt(shape[1]), cb[1])",
                                 "        out = tf.matmul(flattenedInput, g, name='xG')",
                                 "        out = tf.matmul(out, h, name='xGH')",
                                 "    else:",
                                 "        w = self.makeVariable('Weights', shape, 1.0/np.sqrt(shape[0]), cb[0])",
                                 "        out = tf.matmul(flattenedInput, w, name='xW')",
                                 "",
                                 "    if hasBias:",
                                 "        b = self.makeVariable('Biases', [shape[-1]], 0)",
                                 "        out = tf.add(out, b, name=('xWplusB' if len(shape)==2 else 'xGHplusB'))",
                                 "",
                                 "    if l2LossFactor>0:",
                                 "        layerVars = ([w] if len(shape)==2 else [g,h]) + ([b] if hasBias else [])",
                                 "        listOfL2Losses = [ tf.nn.l2_loss(v) for v in layerVars ]",
                                 "        self.l2Loss += l2LossFactor * tf.add_n(listOfL2Losses)",
                                 "",
                                 "    return out",
                                 ""))
                                
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        l2LossFactor = tfBuilder.getL2LossFactor(self)

        tfBuilder.addToGraph( tfBuilder.getScopeStr(self.scope) )
        tfBuilder.graphIndent += 1
        
        if len(self.inShape)==2:
            tfBuilder.addToGraph("seqLen = tf.shape(%s)[1]"%(layerIn))

        if self.rank == 0:
            cb = [ self.netParams[0].codebookSize, 0]
            tfBuilder.addToGraph("out = self.fcLayer(%s, [%d,%d], %s, %s%s)",
                                 (layerIn, self.flatInputSize, self.outSize, str(self.hasBias),
                                  l2LossFactor, (", %s"%cb) if tfBuilder.runQuantized else ""))
        else:
            cb = [ self.netParams[0].codebookSize, self.netParams[1].codebookSize]
            tfBuilder.addToGraph("out = self.fcLayer(%s, [%d,%d,%d], %s, %s%s)",
                                 (layerIn, self.flatInputSize, self.rank, self.outSize, str(self.hasBias),
                                  l2LossFactor, (", %s"%cb) if tfBuilder.runQuantized else ""))

        if len(self.inShape)==2:
            tfBuilder.addToGraph("out = tf.reshape(out, [-1, seqLen, %d])"%(self.outSize))

        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -= 1

# **********************************************************************************************************************
class ConvLayer(Layer):
    argsDic = {
                'k': ('kernel', 'uxu', None),
                's': ('stride', 'uxu', (1,1)),
                'o': ('outDept', 'u', None),
                'p': ('padding', 'p', 'valid'),
                'b': ('hasBias', 'u', 1),
                'd': ('dilation', 'uxu', (1,1)),
                'r': ('rank', 'u', 0),              # Rank=0 => Not Decomposed (Default)
                'l': ('decType', 'dec', 'lr'),      # Current Decomposition types: (Default: lr)
                'e': ('e', 'f', -2.0),              # Only used for LDR
                'f': ('f', 'f', 2.0),               # Only used for LDR
           }
    orderedKeys = 'ksopbdrlef'
    name = 'CONV'
    # ******************************************************************************************************************
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]     # Create a copy
        if self.isVectorIn():
            # In this case the input and output are both vectors
            self.outShape = [self.outDept]
        else:
            assert len(inShape)==3, '%s: Length of input shape to CONV layer must be 3 but it is %d'%(self.scope, len(inShape))
            self.outShape = applyPadding(self.inShape, self.kernel, self.stride, self.padding, self.dilation)
            self.outShape[2] = self.outDept
        
        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )

        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        if self.dilation != (1,1):
            detailsStr = 'KSPD: %s %s %s %s'%(Layer.getArgStr(self.kernel, 'uxu'),
                                              Layer.getArgStr(self.stride, 'uxu'),
                                              Layer.getArgStr(self.padding, 'p'),
                                              Layer.getArgStr(self.dilation, 'uxu'))
        else:
            detailsStr = 'KSP: %s %s %s'%(Layer.getArgStr(self.kernel, 'uxu'),
                                          Layer.getArgStr(self.stride, 'uxu'),
                                          Layer.getArgStr(self.padding, 'p'))
        if self.rank > 0:
            if self.decType == 'lr':    detailsStr += ', LR%d'%(self.rank)
            elif self.decType == 'ldr': detailsStr += ', LDR%d'%(self.rank)
        return detailsStr
    
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        kernelX, kernelY = self.kernel
        inDept = self.inShape[0] if self.isVectorIn() else self.inShape[2]
        
        if self.rank > 0:
            if includeSizeInfo:
                paramStrs = [ '%s/G %dx%dx%dx%d'%(self.scope, kernelY, kernelX, inDept, self.rank),
                              '%s/H 1x1x%dx%d'%(self.scope, self.rank, self.outDept) ]
            else:
                paramStrs = [self.scope + '/G', self.scope + '/H']
        elif includeSizeInfo:
            paramStrs = [ '%s/Weights %dx%dx%dx%d'%(self.scope, kernelY, kernelX, inDept, self.outDept) ]
        else:
            paramStrs = [ self.scope + '/Weights' ]

        if self.hasBias:
            paramStrs += ['%s/Biases%s'%(self.scope, (' %d'%(self.outDept) if includeSizeInfo else ''))]
            
        return paramStrs

    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        listOfL2Losses = [ netParam.tfL2Loss() for netParam in self.netParams ]
        self.l2Loss = factor * tf.add_n(listOfL2Losses)

    # ******************************************************************************************************************
    def inferRank(self, initValues):
        if self.rank > 0:                                       return
        if initValues is None:                                  return
        if initValues[0].shape[3] == self.outDept:              return
        
        assert initValues[0].shape[3] == initValues[1].shape[2],    \
            "%s: LR Init tensor rank mismatch. (%d vs %d)!"%(self.scope, initValues[0].shape[3], initValues[1].shape[2])
        assert initValues[1].shape[3] == self.outDept,              \
            "%s: LR Init tensor outDept mismatch! (%d vs %d)"%(self.scope, initValues[1].shape[3], self.outDept)
        self.rank = initValues[1].shape[2]

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        kernelX, kernelY = self.kernel
        inDept = self.inShape[-1]
        self.inferRank(initValues)
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            if initValues is None:
                if self.rank == 0:
                    initW = tfr.truncated_normal([kernelY, kernelX, inDept, self.outDept], mean=0,
                                                 stddev=1.0/np.sqrt(kernelY*kernelX*inDept), seed=SEED)
                else:
                    assert self.decType=='lr', "%s: Only LR Decomposition is supported for Conv Layers!"
                    initG = tfr.truncated_normal([kernelY, kernelX, inDept, self.rank], mean=0,
                                                 stddev=1.0/np.sqrt(kernelY*kernelX*inDept), seed=SEED)
                    initH = tfr.truncated_normal([1, 1, self.rank, self.outDept], mean=0,
                                                 stddev=1.0/np.sqrt(self.rank), seed=SEED)
                if self.hasBias:    initB = tf.constant(0.0, shape=[self.outDept])
            else:
                if self.rank > 0:
                    assert self.decType=='lr', "%s: Only LR Decomposition is supported for Conv Layers!"
                    initG, initH = initValues[0], initValues[1]
                    self.checkShape('G', initG.shape, (kernelY, kernelX, inDept, self.rank))
                    self.checkShape('H', initH.shape, (1, 1, self.rank, self.outDept))
                else:
                    initW = initValues[0]
                    self.checkShape('Weights', initW.shape, (kernelY, kernelX, inDept, self.outDept))

                if self.hasBias:
                    initB = initValues[2] if self.rank>0 else initValues[1]
                    self.checkShape('Biases', initB.shape, (self.outDept,))

            self.netParams = []
            if self.rank > 0:
                self.g = self.makeNetParam('G', initG)
                self.h = self.makeNetParam('H', initH)
            else:
                self.weights = self.makeNetParam('Weights',initW)

            if self.hasBias:
                self.biases = self.makeNetParam('Biases',initB)
                
        return self.netParams
    
    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        kernelX, kernelY = self.kernel
        strideX, strideY = self.stride
        dilationX, dilationY = self.dilation
        inDept = self.inShape[-1]
        outputs = []
        with tf.name_scope(self.scope):
            if self.isVectorIn():
                assert (kernelX==1 and strideX==1), "%s: Feeding vector to %s layer, Kernel and Stride must be 1!"%(self.scope, self.name)
                # If the input is Vector and kernel and stride are 1, we make this as a 1x1 convolution
                # which is applied on a 1x1 image with depth equal to length of vector.
                # This is specifically used in MobileNet
                # In this case the input and output are both vectors
                reshapedIn = tf.reshape(input, [-1, 1, 1, self.inShape[0]], name='Make1x1Img')
            else:
                reshapedIn = input

            if self.padding not in ['valid', 'same']:
                conv = tf.nn.conv2d(tf.pad(reshapedIn, self.get4dPadding(), name='Pad'),
                                    self.weights if self.rank==0 else self.g,
                                    strides=[1, strideY, strideX, 1],
                                    padding='VALID',
                                    dilations=(dilationY, dilationX),
                                    name='xW' if self.rank==0 else 'xG')
            else:
                conv = tf.nn.conv2d(reshapedIn,
                                    self.weights if self.rank==0 else self.g,
                                    strides=[1, strideY, strideX, 1],
                                    padding=self.padding.upper(),
                                    dilations=(dilationY, dilationX),
                                    name='xW' if self.rank==0 else 'xG')
            if self.rank > 0:
                conv = tf.nn.conv2d(conv, self.h, strides=[1, 1, 1, 1], padding='VALID', name='xGH')

            if self.hasBias:    outputs += [ tf.add(conv, self.biases, name='xWplusB' if self.rank==0 else 'xGHplusB') ]
            else:               outputs += [ conv ]

            if self.isVectorIn():
                # Reshape the output back to Vectors
                conv = tf.reshape(conv, [-1, self.outDept], name='Flatten')
            
            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None
 
    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        kernelX, kernelY = self.kernel
        strideX, strideY = self.stride
        dilationX, dilationY = self.dilation

        if self.isVectorIn():
            # See "buildGraph" above for more details about this.
            inputName = onnxBuilder.addReshape(inputName, [-1,self.inShape[0],1,1], s+'Input4D')

        params = self.npNetParams
        wName, gName, hName, bName = ((s+x) for x in ['Weights', 'G', 'H', 'Biases'])
        bNameList = [bName] if self.hasBias else []

        padding = self.padding
        if ((dilationX != 1) or (dilationY != 1)) and padding=='same':
            # ONNX does not support dilation with same padding!!!
            # We calculate the padding needed to make the output dimensions the same as the input
            assert len(self.inShape)==3
            y, x, _ = self.inShape
            # Dims without padding:
            xnp = int(np.ceil( (x - (kernelX-1)*dilationX)/float(strideX) ))
            ynp = int(np.ceil( (y - (kernelY-1)*dilationY)/float(strideY) ))
            # Expected dims with "same" padding:
            xsp = x//strideX
            ysp = y//strideY
            
            # In case of odd values, add extra padding to the right and buttom
            left = (xsp - xnp)*strideX//2
            right = (xsp - xnp)*strideX - left

            top = (ysp - ynp)*strideY//2
            bottom = (ysp - ynp)*strideY - top

            padding = [ [top, bottom], [left, right] ]
            
        if self.rank > 0:
            onnxBuilder.addNetParam(gName, params[0])
            onnxBuilder.addNetParam(hName, params[1])

            onnxBuilder.addConv([inputName, gName], [s+'xG'], s+'ConvG',
                                [kernelY,kernelX], [strideY,strideX], [dilationY,dilationX], padding)

            # Using the default values for the attributes: alpha=1, beta=1, transA=0, transB=0
            outputName = s+('xGHplusB' if self.hasBias else 'xGH')
            onnxBuilder.addConv([s+'xG', hName] + bNameList, [outputName], s+'ConvH')
        else:
            onnxBuilder.addNetParam(wName, params[0])
            outputName = s+('xWplusB' if self.hasBias else 'xW')
            onnxBuilder.addConv([inputName, wName] + bNameList, [outputName], s+'Conv',
                                [kernelY,kernelX], [strideY,strideX], [dilationY,dilationX], padding)

        if self.hasBias:
            onnxBuilder.addNetParam(bName, params[-1])

        if self.isVectorIn():
            # Reshape the output back to Vectors
            outputName = onnxBuilder.addReshape(outputName, [-1,self.outDept])

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        inDepth = self.inShape[-1]

        params = NetParam.toNp(self.netParams, self.model.session)
        if self.rank > 0:
            g, h, b = params if self.hasBias else (params[0], params[1], None)
            cmlBuilder.addConv(s+'ConvG', inputName, s+'xG', inDepth, self.rank, self.kernel, self.stride,
                               self.dilation, self.padding, g, None)

            outputName = s+('xGHplusB' if self.hasBias else 'xGH')
            cmlBuilder.addConv(s+'ConvH', s+'xG', outputName, self.rank, self.outDept, (1,1), (1,1),
                               (1,1), 'valid', h, b)

        else:
            w, b = params if self.hasBias else (params[0], None)
            outputName = s+('xWplusB' if self.hasBias else 'xW')
            cmlBuilder.addConv(s+'Conv', inputName, outputName, inDepth, self.outDept, self.kernel, self.stride,
                               self.dilation, self.padding, w, b)

        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('convLayer'):
            tfBuilder.defineMethod('convLayer')
            tfBuilder.addMethod(("def convLayer(self, layerIn, shape, strides, padding, dilations, hasBias, l2LossFactor, cb=[0,0]):",
                                 "    if len(shape)==5:",
                                 "        g = self.makeVariable('G', shape[:-1], 1.0/np.sqrt(np.prod(shape[:3])), cb[0])",
                                 "        h = self.makeVariable('H', [1,1]+shape[3:], 1.0/np.sqrt(shape[2]), cb[1])",
                                 "    else:",
                                 "        w = self.makeVariable('Weights', shape, 1.0/np.sqrt(np.prod(shape[:3])), cb[0])",
                                 "",
                                 "    out = tf.nn.conv2d(layerIn, w if len(shape)==4 else g,",
                                 "                       strides=[1, strides[0], strides[1], 1],",
                                 "                       padding=padding,",
                                 "                       dilations=dilations,",
                                 "                       name='xW' if len(shape)==4 else 'xG')",
                                 "    if len(shape)==5:",
                                 "        out = tf.nn.conv2d(out, h, strides=[1, 1, 1, 1], padding='VALID', name='xGH')",
                                 "",
                                 "    if hasBias:",
                                 "        b = self.makeVariable('Biases', [shape[-1]], 0)",
                                 "        out = tf.add(out, b, name='xWplusB' if len(shape)==4 else 'xGHplusB')",
                                 "",
                                 "    if l2LossFactor>0:",
                                 "        layerVars = ([w] if len(shape)==4 else [g,h]) + ([b] if hasBias else [])",
                                 "        listOfL2Losses = [ tf.nn.l2_loss(v) for v in layerVars ]",
                                 "        self.l2Loss += l2LossFactor * tf.add_n(listOfL2Losses)",
                                 "",
                                 "    return out",
                                 ""))
                                 
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        l2LossFactor = tfBuilder.getL2LossFactor(self)
        tfBuilder.addToGraph( tfBuilder.getScopeStr(self.scope) )
        tfBuilder.graphIndent += 1
        
        if self.isVectorIn():
            tfBuilder.addToGraph("out = tf.reshape(%s, [-1, 1, 1, %d], name='Make1x1Img')",
                                 (layerIn, self.inShape[0]))
            layerIn = 'out'
                                
        if self.padding not in ['valid', 'same']:
            tfBuilder.addToGraph("out = tf.pad(%s, %s, name='Pad')"%(layerIn, str(self.get4dPadding())))
            padding = 'VALID'
            layerIn = 'out'
        else:
            padding = self.padding.upper()
                                 
        if self.rank == 0:
            cb = [ self.netParams[0].codebookSize, 0]
            tfBuilder.addToGraph("out = self.convLayer(%s, [%d,%d,%d,%d], %s, '%s', %s, %s, %s%s)",
                                 (layerIn, self.kernel[1], self.kernel[0], self.inShape[-1], self.outDept,
                                  str(self.stride[::-1]), padding, str(self.dilation[::-1]),
                                  str(self.hasBias), l2LossFactor, (", %s"%cb) if tfBuilder.runQuantized else ""))
        else:
            cb = [ self.netParams[0].codebookSize, self.netParams[1].codebookSize]
            tfBuilder.addToGraph("out = self.convLayer(%s, [%d,%d,%d,%d,%d], %s, '%s', %s, %s, %s%s)",
                                 (layerIn, self.kernel[1], self.kernel[0], self.inShape[-1], self.rank, self.outDept,
                                  str(self.stride[::-1]), padding, str(self.dilation[::-1]),
                                  str(self.hasBias), l2LossFactor, (", %s"%cb) if tfBuilder.runQuantized else ""))

        if self.isVectorIn():
            # Reshape the output back to Vectors
            tfBuilder.addToGraph("out = tf.reshape(out, [-1, %d], name='Flatten')"%(self.outDept))
                                
        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -= 1

# **********************************************************************************************************************
class DwConvLayer(Layer):     # Depth-wise Convolutional
    argsDic = {
                'k': ('kernel', 'uxu', None),
                's': ('stride', 'uxu', (1,1)),
                'p': ('padding', 'p', 'valid'),
                'b': ('hasBias', 'u', 1),
                'r': ('rank', 'u', 0),              # Rank=0 => Not Decomposed (Default)
                'l': ('decType', 'dec', 'lr'),      # Current Decomposition types: (Default: lr)
                'e': ('e', 'f', -2.0),              # Only used for LDR
                'f': ('f', 'f', 2.0),               # Only used for LDR
              }
    orderedKeys = 'kspbrlef'
    name = 'DWCN'
    num2Factors = { 3: { 128: (36, 32), 256: (48, 48), 512: (72, 64), 1024: (96, 96), 576: (72, 72), 960: (90, 96)},
                    5: { 128: (50, 64), 256: (80, 80), 512: (100, 128), 1024: (160, 160)} }
    # ******************************************************************************************************************
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]
        self.outShape = applyPadding(inShape, self.kernel, self.stride, self.padding)
        
        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )

        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        detailsStr = 'KSP: %s %s %s'%(Layer.getArgStr(self.kernel, 'uxu'),
                                      Layer.getArgStr(self.stride, 'uxu'),
                                      Layer.getArgStr(self.padding, 'p'))
        if self.rank > 0:
            if self.decType == 'lr':    detailsStr += ', LR%d'%(self.rank)
            elif self.decType == 'ldr': detailsStr += ', LDR%d'%(self.rank)

        return detailsStr
    
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        kernelX, kernelY = self.kernel
        inDept = self.inShape[-1]
        
        if self.rank > 0:
            if includeSizeInfo:
                paramStrs = [ '%s/G %dx%d'%(self.scope, kernelX*kernelY, self.rank),
                              '%s/H %dx%d'%(self.scope, self.rank, inDept) ]
            else:
                paramStrs = [self.scope + '/G', self.scope + '/H']
        elif includeSizeInfo:
            paramStrs = [ '%s/Weights %dx%dx%dx1'%(self.scope, kernelY, kernelX, inDept) ]
        else:
            paramStrs = [ self.scope + '/Weights' ]

        if self.hasBias:
            paramStrs += ['%s/Biases%s'%(self.scope, (' %d'%(inDept) if includeSizeInfo else ''))]
            
        return paramStrs
    
    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        listOfL2Losses = [ netParam.tfL2Loss() for netParam in self.netParams ]
        self.l2Loss = factor * tf.add_n(listOfL2Losses)

    # ******************************************************************************************************************
    def inferRank(self, initValues):
        inDept = self.inShape[-1]
        kernelX, kernelY = self.kernel
        if self.rank > 0:                                           return
        if initValues is None:                                      return
        if len(initValues[0].shape)==4:                             return
        assert len(initValues[0].shape)==2,                         \
            "%s: Invalid init tensor dimension (%d vs 2)!"%(self.scope, len(initValues[0].shape))
        assert initValues[0].shape[0] == kernelY*kernelX,           \
            "%s: Invalid init tensor size (%d vs %d)!"%(self.scope, initValues[0].shape[0], kernelY*kernelX)
        assert initValues[0].shape[1] == initValues[1].shape[0],    \
            "%s: LR Init tensor rank mismatch (%d vs %d)!"%(self.scope, initValues[0].shape[1], initValues[1].shape[0])
        assert initValues[1].shape[1] == inDept,                    \
            "%s: LR Init tensor inDept mismatch (%d vs %d)!"%(self.scope, initValues[1].shape[1], inDept)
        self.rank = initValues[0].shape[1]

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        kernelX, kernelY = self.kernel
        inDept = self.inShape[-1]
        self.inferRank(initValues)
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            if initValues is None:
                if self.rank == 0:
                    initW = tfr.truncated_normal([kernelY, kernelX, inDept, 1], mean=0,
                                                 stddev=1.0/np.sqrt(kernelY*kernelX*inDept), seed=SEED)
                else:
                    assert self.decType=='lr', "%s: Only LR Decomposition is supported for Depth-wise Conv Layers!"%(self.scope)
                    initG = tfr.truncated_normal([kernelY*kernelX, self.rank], mean=0,
                                                 stddev=1.0/np.sqrt(kernelY*kernelX), seed=SEED)
                    initH = tfr.truncated_normal([self.rank, inDept], mean=0,
                                                 stddev=1.0/np.sqrt(self.rank), seed=SEED)
                if self.hasBias:    initB = tf.constant(0.0, shape=[inDept])
            else:
                if self.rank > 0:
                    assert self.decType=='lr', "%s: Only LR Decomposition is supported for Depth-wise Conv Layers!"%(self.scope)
                    initG, initH = initValues[0], initValues[1]
                    self.checkShape('G', initG.shape, (kernelY*kernelX, self.rank))
                    self.checkShape('H', initH.shape, (self.rank, inDept))
                else:
                    initW = initValues[0]
                    self.checkShape('Weights', initW.shape, (kernelY, kernelX, inDept, 1))

                if self.hasBias:
                    initB = initValues[2] if self.rank>0 else initValues[1]
                    self.checkShape('Biases', initB.shape, (inDept,))

            self.netParams = []
            if self.rank > 0:
                self.g = self.makeNetParam('G', initG)
                self.h = self.makeNetParam('H', initH)
            else:
                self.weights = self.makeNetParam('Weights',initW)

            if self.hasBias:
                self.biases = self.makeNetParam('Biases', initB)
                
        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        kernelX, kernelY = self.kernel
        strideX, strideY = self.stride
        outputs = []
        inDept = self.inShape[-1]
        with tf.name_scope(self.scope):
            if self.rank > 0:   weights = tf.reshape( tf.matmul(self.g, self.h), [kernelY, kernelX, inDept, 1], 'GH')
            else:               weights = self.weights
            
            if self.padding not in ['valid', 'same']:
                conv = tf.nn.depthwise_conv2d(tf.pad(input, self.get4dPadding(), name='Pad'),
                                              weights,
                                              strides=[1, strideY, strideX, 1],
                                              padding='VALID',
                                              name='xW' if self.rank==0 else 'xGH')
            else:
                conv = tf.nn.depthwise_conv2d(input,
                                              weights,
                                              strides=[1, strideY, strideX, 1],
                                              padding=self.padding.upper(),
                                              name='xW' if self.rank==0 else 'xGH')

            if self.hasBias:    outputs += [ tf.add(conv, self.biases, name='xWplusB' if self.rank==0 else 'xGHplusB') ]
            else:               outputs += [ conv ]

            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        kernelX, kernelY = self.kernel
        strideX, strideY = self.stride
        inDept = self.inShape[-1]

        params = self.npNetParams
        wName, bName = ((s+x) for x in ['Weights', 'Biases'])
        bNameList = [bName] if self.hasBias else []
 
        if self.rank > 0:
            # Low-Rank Depth-wise Convolutions is not recommended when exporting to ONNX. Here we create a "w"
            # by running a matrix multiplication on "g" and "h". This cancels all the gains of low-rank decomposition and
            # quantization. If we didn't use low-rank, at least we could keep the quantization here.
            myPrint("Warning(%s): Low-Rank Depth-wise Convolutions is not recommended when exporting to ONNX!", color='yellow')
            g, h = params[0], params[1]
            w = g.value().dot(h.value()).reshape( (kernelY, kernelX, inDept, 1) )
            onnxBuilder.addNetParam(wName, NetParam('NP', w))
        else:
            onnxBuilder.addNetParam(wName, params[0])
            
        outputName = s + ('xWplusB' if self.hasBias else 'xW')
        onnxBuilder.addConv([inputName, wName] + bNameList, [outputName], s+'DwConv',
                            [kernelY,kernelX], [strideY,strideX], [1,1], self.padding, inDept)

        if self.hasBias:
            onnxBuilder.addNetParam(bName, params[-1])

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        params = NetParam.toNp(self.netParams, self.model.session)
        if self.rank > 0:
            # Low-Rank Depth-wise Convolutions is not recommended when exporting to CoreML. Here we create a "w"
            # by running a matrix multiplication on "g" and "h". This cancels all the gains of low-rank decomposition and
            # quantization. If we don't use low-rank, at least we can keep the quantization here.
            myPrint("Warning(%s): Low-Rank Depth-wise Convolutions is not recommended when exporting to CoreML!", color='yellow')
            g, h, b = params if self.hasBias else (params[0], params[1], None)
            w = g.value().dot(h.value()).reshape( (self.kernel[1], self.kernel[0], self.inShape[-1], 1) )
            w = NetParam(w)
        else:
            w, b = params if self.hasBias else (params[0], None)
            
        outputName = s + ('xWplusB' if self.hasBias else 'xW')
        # Use outDept=0 to tell this is a Depthwise Conv.
        cmlBuilder.addConv(s+'DwConv', inputName, outputName, self.inShape[-1], 0, self.kernel, self.stride, (1,1),
                           self.padding, w, b)
                                  
        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('dwConvLayer'):
            tfBuilder.defineMethod('dwConvLayer')
            tfBuilder.addMethod(("def dwConvLayer(self, layerIn, shape, strides, padding, hasBias, l2LossFactor, cb=[0,0]):",
                                 "    # shape: [kernelY, kernelX, [rank], inDept] ",
                                 "    if len(shape)==4:  # Low-Rank",
                                 "        g = self.makeVariable('G', [shape[0]*shape[1], shape[2]], 1.0/np.sqrt(np.prod(shape[:2])), cb[0])",
                                 "        h = self.makeVariable('H', shape[2:], 1.0/np.sqrt(shape[2]), cb[1])",
                                 "        w = tf.reshape( tf.matmul(g, h), shape[:2]+[shape[3],1], 'GH')",
                                 "    else:",
                                 "        w = self.makeVariable('Weights', shape+[1], 1.0/np.sqrt(np.prod(shape)), cb[0])",
                                 "",
                                 "    out = tf.nn.depthwise_conv2d(layerIn, w, strides=[1, strides[0], strides[1], 1],",
                                 "                                 padding=padding, name='xW' if len(shape)==3 else 'xGH')",
                                 "",
                                 "    if hasBias:",
                                 "        b = self.makeVariable('Biases', [shape[-1]], 0)",
                                 "        out = tf.add(out, b, name='xWplusB' if len(shape)==3 else 'xGHplusB')",
                                 "",
                                 "    if l2LossFactor>0:",
                                 "        layerVars = ([w] if len(shape)==3 else [g,h]) + ([b] if hasBias else [])",
                                 "        listOfL2Losses = [ tf.nn.l2_loss(v) for v in layerVars ]",
                                 "        self.l2Loss += l2LossFactor * tf.add_n(listOfL2Losses)",
                                 "",
                                 "    return out",
                                 ""))
                                 
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        l2LossFactor = tfBuilder.getL2LossFactor(self)
        tfBuilder.addToGraph( tfBuilder.getScopeStr(self.scope) )
        tfBuilder.graphIndent += 1
                                 
        if self.padding not in ['valid', 'same']:
            tfBuilder.addToGraph("out = tf.pad(%s, %s, name='Pad')"%(layerIn, str(self.get4dPadding())))
            padding = 'VALID'
            layerIn = 'out'
        else:
            padding = self.padding.upper()
                                 
        if self.rank == 0:
            cb = [ self.netParams[0].codebookSize, 0]
            tfBuilder.addToGraph("out = self.dwConvLayer(%s, [%d,%d,%d], %s, '%s', %s, %s%s)",
                                 (layerIn, self.kernel[1], self.kernel[0], self.inShape[-1],
                                  str(self.stride[::-1]), padding, str(self.hasBias), l2LossFactor,
                                  (", %s"%cb) if tfBuilder.runQuantized else ""))
        else:
            cb = [ self.netParams[0].codebookSize, self.netParams[1].codebookSize] if runQuantized else [0,0]
            tfBuilder.addToGraph("out = self.dwConvLayer(%s, [%d,%d,%d,%d], %s, '%s', %s, %s%s)",
                                 (layerIn, self.kernel[1], self.kernel[0], self.rank, self.inShape[-1],
                                  str(self.stride[::-1]), padding, str(self.hasBias), l2LossFactor,
                                  (", %s"%cb) if tfBuilder.runQuantized else ""))
                                
        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -= 1
        
# **********************************************************************************************************************
class BnLayer(Layer):
    argsDic = {
                'e': ('epsilon', 'f', 0.001)
              }
    orderedKeys = 'e'
    name = 'BN'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        
    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]
        self.outShape = [x for x in inShape]
        
        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )
        
        return self.outShape
    
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        if includeSizeInfo:
            paramSize = self.inShape[-1]
            return ['%s/Beta %d'%(self.scope, paramSize),
                    '%s/Gamma %d'%(self.scope, paramSize),
                    '%s/MovingMean %d'%(self.scope, paramSize),
                    '%s/MovingVar %d'%(self.scope, paramSize)]
        return [self.scope + '/Beta', self.scope + '/Gamma', self.scope + '/MovingMean', self.scope + '/MovingVar']
    
    # ******************************************************************************************************************
    def makeVars(self, initValues):
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            inDept = self.inShape[-1]
            if initValues is None:
                initB = tf.constant(0.0, shape=[inDept])
                initG = tf.constant(1.0, shape=[inDept])
                initM = tf.constant(0.0, shape=[inDept])
                initV = tf.constant(1.0, shape=[inDept])
            else:
                initB, initG, initM, initV  = initValues[0:4]
                self.checkShape('Beta', initB.shape, (inDept,))
                self.checkShape('Gamma', initG.shape, (inDept,))
                self.checkShape('MovingMean', initM.shape, (inDept,))
                self.checkShape('MovingVar', initV.shape, (inDept,))

            self.netParams = []
            self.beta = self.makeNetParam('Beta', initB)
            self.gamma = self.makeNetParam('Gamma', initG)
            self.movingMean = self.makeNetParam('MovingMean', initM, trainable=False)
            self.movingVar = self.makeNetParam('MovingVar', initV, trainable=False)
            
        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        # The following website helped me implement this:
        #     https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        outputs = []
        moments = None
        with tf.name_scope(self.scope):
            if isTraining:
                batchMean, batchVar = tf.nn.moments(input, [0,1,2], keep_dims=False, name='BatchMeanAndVar')
                outputs += [ tf.nn.batch_normalization(input, batchMean, batchVar, self.beta, self.gamma,
                                                       self.epsilon, name='BatchNorm') ]
                moments = [ (batchMean, self.movingMean), (batchVar, self.movingVar) ]
            else:
                outputs += [ tf.nn.batch_normalization(input, self.movingMean, self.movingVar, self.beta, self.gamma,
                                                       self.epsilon, name='BatchNorm') ]

            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], moments

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        params = self.npNetParams
        bName, gName, mName, vName = ((s+x) for x in ['Beta', 'Gamma', 'MovingMean', 'MovingVar'])

        onnxBuilder.addNetParam(bName, params[0])
        onnxBuilder.addNetParam(gName, params[1])
        onnxBuilder.addNetParam(mName, params[2])
        onnxBuilder.addNetParam(vName, params[3])

        outputName = s[:-1]
        onnxBuilder.addNode('BatchNormalization', [inputName, gName, bName, mName, vName], [outputName],
                            s+'BatchNormalization', epsilon=self.epsilon)

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        b, g, m, v = NetParam.toNp(self.netParams, self.model.session)
        outputName = s[:-1]
        cmlBuilder.add_batchnorm(name =                    s+'batchnorm',
                                 channels =                self.inShape[-1],
                                 gamma =                   g.value(),
                                 beta =                    b.value(),
                                 mean =                    m.value(),
                                 variance =                v.value(),
                                 input_name =              inputName,
                                 output_name =             outputName,
                                 compute_mean_var =        False,
                                 instance_normalization =  False,
                                 epsilon =                 self.epsilon)

        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)
        
    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('bnLayer'):
            tfBuilder.defineMethod('bnLayer')
            tfBuilder.addMethod(("def bnLayer(self, layerIn, shape, epsilon, isTraining):",
                                 "    beta = self.makeVariable('Beta', shape, 0)",
                                 "    gamma = self.makeVariable('Gamma', shape, 1)",
                                 "    movingMean = self.makeVariable('MovingMean', shape, 0, trainable=False)",
                                 "    movingVar = self.makeVariable('MovingVar', shape, 1, trainable=False)",
                                 "",
                                 "    if isTraining:",
                                 "        batchMean, batchVar = tf.nn.moments(layerIn, [0,1,2], keep_dims=False, name='BatchMeanAndVar')",
                                 "        out = tf.nn.batch_normalization(layerIn, batchMean, batchVar, beta, gamma, epsilon, name='BatchNorm')",
                                 "        moments = [ (batchMean, movingMean), (batchVar, movingVar) ]",
                                 "    else:",
                                 "        out = tf.nn.batch_normalization(layerIn, movingMean, movingVar, beta, gamma, epsilon, name='BatchNorm')",
                                 "",
                                 "    return out",
                                 ""))
        
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                              "    out = self.bnLayer(%s, [%d], %s, isTraining)"),
                             (layerIn, self.inShape[-1], str(self.epsilon)))
        
        tfBuilder.graphIndent +=1
        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -=1

# **********************************************************************************************************************
class LnLayer(Layer):
    argsDic = {
                'e': ('epsilon', 'f', 1.0e-12)
              }
    orderedKeys = 'e'
    name = 'LN'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        
    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]
        self.outShape = [x for x in inShape]
        
        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )
        
        return self.outShape

    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        paramSize = self.inShape[-1]
        if includeSizeInfo:
            return ['%s/Beta %d'%(self.scope, paramSize),
                    '%s/Gamma %d'%(self.scope, paramSize)]
        return [self.scope + '/Beta', self.scope + '/Gamma']
    
    # ******************************************************************************************************************
    def makeVars(self, initValues):
        paramSize = self.inShape[-1]
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            if initValues is None:
                initB = tf.constant(0.0, shape=[paramSize])
                initG = tf.constant(1.0, shape=[paramSize])
            else:
                initB, initG  = initValues[0:2]
                self.checkShape('Beta', initB.shape, (paramSize,))
                self.checkShape('Gamma', initG.shape, (paramSize,))

            self.netParams = []
            self.beta = self.makeNetParam('Beta', initB)
            self.gamma = self.makeNetParam('Gamma', initG)
            
        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        outputs = []
        moments = None
        with tf.name_scope(self.scope):
            mean, variance = tf.nn.moments(input, [-1], keep_dims=True, name='MeanAndVar')
            outputs += [ tf.nn.batch_normalization(input, mean, variance, self.beta, self.gamma,
                                                   self.epsilon, name='LayerNorm') ]

            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        params = self.npNetParams
        bName, gName, eName = ((s+x) for x in ['Beta', 'Gamma', 'Epsilon'])
        
        if len(self.inShape)==1:    bgShape = [1, self.inShape[-1]]     # LN following FC
        elif len(self.inShape)==2:  bgShape = [1, 1, self.inShape[-1]]  # LN following BERT or EMB
        else:                       bgShape = [1,self.inShape[-1],1,1]  # LN following Conv.

        onnxBuilder.addNetParam(bName, params[0], bgShape)
        onnxBuilder.addNetParam(gName, params[1], bgShape)
        onnxBuilder.addParam(eName, 'float', [], [self.epsilon])

        ax = [1] if len(self.inShape)==3 else [-1]
        outputName = s+'LN'
        onnxBuilder.addNode('ReduceMean', [inputName], [s+'u'], s+'ReduceMean1', axes=ax)
        onnxBuilder.addNode('Sub', [inputName, s+'u'], [s+'x-u'], s+'Sub')
        onnxBuilder.addNode('Mul', [s+'x-u', s+'x-u'], [s+'(x-u)2'], s+'Mul1')
        onnxBuilder.addNode('ReduceMean', [s+'(x-u)2'], [s+'sigma2'], s+'ReduceMean2', axes=ax)
        onnxBuilder.addNode('Add', [s+'sigma2', eName], [s+'sigma2+epsilon'], s+'Add1')
        onnxBuilder.addNode('Sqrt', [s+'sigma2+epsilon'], [s+'sigma'], s+'Sqrt')
        onnxBuilder.addNode('Mul', [gName, s+'x-u'], [s+'gamma*(x-u)'], s+'Mul2')
        onnxBuilder.addNode('Div', [s+'gamma*(x-u)', s+'sigma'], [s+'LN-Beta'], s+'Div')
        onnxBuilder.addNode('Add', [s+'LN-Beta', bName], [outputName], s+'Add2')

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        # For Bert, the shape of input and output is: seqLen, 1, paramSize, 1, 1
        paramSize = self.inShape[-1]
        b, g = NetParam.toNp(self.netParams, self.model.session)

        outputName = s + 'mvn'
        cmlBuilder.add_mvn(outputName, inputName, outputName, across_channels=True, normalize_variance=True,
                           epsilon=self.epsilon)
        inputName, outputName = outputName, s + 'scale'
        cmlBuilder.add_scale(outputName, g.value(), b.value(), True, inputName, outputName,
                             shape_scale=[paramSize,1,1], shape_bias=[paramSize,1,1])

        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('lnLayer'):
            tfBuilder.defineMethod('lnLayer')
            tfBuilder.addMethod(("def lnLayer(self, layerIn, shape, epsilon):",
                                 "    beta = self.makeVariable('Beta', shape, 0)",
                                 "    gamma = self.makeVariable('Gamma', shape, 1)",
                                 "",
                                 "    mean, variance = tf.nn.moments(layerIn, [-1], keep_dims=True, name='MeanAndVar')",
                                 "    return tf.nn.batch_normalization(layerIn, mean, variance, beta, gamma, epsilon, name='LayerNorm')",
                                 ""))
        
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                              "    out = self.lnLayer(%s, [%d], %s)"),
                             (layerIn, self.inShape[-1], str(self.epsilon)))
        
        tfBuilder.graphIndent +=1
        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -=1
        

# **********************************************************************************************************************
class BertLayer(Layer):
    # Useful links for Transformers:
    #    https://www.tensorflow.org/tutorials/text/transformer
    argsDic = {
                'o': ('outSize', 'u', None),
                'i': ('intermediateSize', 'u', None),
                'h': ('numHeads', 'u', 12),     # Num Attention Heads
                'r': ('dropRate', 'f', 0.1),    # Dropout rate (Probability of dropping)
                's': ('initStd', 'f', 0.02),    # Standard Dev. for initializers
                'e': ('epsilon', 'f', 1.0e-12)  # epsilon for the normalization after embedding
              }
    orderedKeys = 'oihrse'
    name = 'BERT'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        
        self.headMask = None
        assert (self.outSize%self.numHeads)==0, "outSize must be a multiple of numHeads!"
        
        subLayerInfo = [ ('SelfQuery',      'FC_O%d:None'%( self.outSize )),
                         ('SelfKey',        'FC_O%d:None'%( self.outSize )),
                         ('SelfValue',      'FC_O%d:None'%( self.outSize )),
                         ('SelfOut',        'FC_O%d:None:DO_R%f'%( self.outSize, self.dropRate )),
                         ('SelfNorm',       'LN_E%f'%(self.epsilon)),
                         ('Intermediate',   'FC_O%d:%s'%( self.intermediateSize, actStr )),
                         ('Out',            'FC_O%d:None:DO_R%f'%( self.outSize, self.dropRate )),
                         ('OutNorm',        'LN_E%f'%(self.epsilon)) ]

        self.queryFc, self.keyFc, self.valueFc, self.selfFc, self.selfLn, self.intermediateFc, self.outFc, self.outLn =\
            [self.layers.getLayer(l, scope=x[0], layerStr=x[1], parent=self) for l,x in enumerate(subLayerInfo)]

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        # Shape IO:
        #    bert: [-1, outSize] -> [-1, outSize]           (seqLen can vary)
        #    CNN:  [height, width, outSize] -> [height*width, outSize]
        self.inShape = [x for x in inShape]
        if len(inShape)==3:
            # Bert following CNN:
            self.outShape = [inShape[0]*inShape[1], inShape[2]]
        else:
            assert inShape[1] == self.outSize, "%s: Input size (%d) does not match output size (%d)!"%(self.scope, inShape[1], self.outSize)
            self.outShape = [x for x in inShape]

        # Set internal layers input/output shapes:
        internalShapes = [inShape[-1]]
        for layer in [self.queryFc, self.keyFc, self.valueFc, self.selfFc, self.selfLn, self.intermediateFc, self.outLn]:
            layer.getOutShape(internalShapes)
        self.outFc.getOutShape([self.intermediateSize])
        
        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )
            
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        shortStr = '%d/%d, %d heads'%(self.outSize, self.intermediateSize, self.numHeads)
        return shortStr
    
    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        paramStrs = []
        
        strs  = self.queryFc.getAllParamStrs(includeSizeInfo)
        strs += self.keyFc.getAllParamStrs(includeSizeInfo)
        strs += self.valueFc.getAllParamStrs(includeSizeInfo)
        strs += self.selfFc.getAllParamStrs(includeSizeInfo)
        strs += self.selfLn.getAllParamStrs(includeSizeInfo)
        paramStrs += [ '%s/SelfAttention/%s'%(self.scope,ps) for ps in strs ]

        strs = self.intermediateFc.getAllParamStrs(includeSizeInfo)
        paramStrs += [ '%s/Intermediate/%s'%(self.scope,ps) for ps in strs ]

        strs  = self.outFc.getAllParamStrs(includeSizeInfo)
        strs += self.outLn.getAllParamStrs(includeSizeInfo)
        paramStrs += [ '%s/Output/%s'%(self.scope,ps) for ps in strs ]
            
        return paramStrs

    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        listOfL2Losses = [ netParam.tfL2Loss() for netParam in self.netParams ]
        self.l2Loss = factor * tf.add_n(listOfL2Losses)

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.netParams = []
        initIndex = 0
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            with tf.name_scope('SelfAttention'), tf1.variable_scope('SelfAttention',reuse=tf1.AUTO_REUSE):
                for layer in [self.queryFc, self.keyFc, self.valueFc, self.selfFc, self.selfLn]:
                    layerVars = layer.makeVars(None if initValues is None else initValues[initIndex:])
                    initIndex += len(layerVars)
                    self.netParams += layerVars

            with tf.name_scope('Intermediate'), tf1.variable_scope('Intermediate',reuse=tf1.AUTO_REUSE):
                layerVars = self.intermediateFc.makeVars(None if initValues is None else initValues[initIndex:])
                initIndex += len(layerVars)
                self.netParams += layerVars

            with tf.name_scope('Output'), tf1.variable_scope('Output',reuse=tf1.AUTO_REUSE):
                for layer in [self.outFc, self.outLn]:
                    layerVars = layer.makeVars(None if initValues is None else initValues[initIndex:])
                    initIndex += len(layerVars)
                    self.netParams += layerVars

        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        outputs = []
        
        def permute(x, numHeads, seqLen, headSize):
            x = tf.reshape(x, [-1, seqLen, numHeads, headSize])
            return tf.transpose(x, perm=[0, 2, 1, 3])
            
        tfInShape = tf.shape(input)
        if len(self.inShape)==3:    seqLen = tfInShape[1]*tfInShape[2]      # Bert following CNN:
        else:                       seqLen = tfInShape[1]
        headSize = self.outSize//self.numHeads
        with tf.name_scope(self.scope):
            input = tf.reshape(input, (-1, self.outSize))           # Shape: [batch*seqLen, outSize]
            with tf.name_scope('SelfAttention'):
                query,_ = self.queryFc.buildGraph(input, isTraining)  # Shape: [batch*seqLen, outSize]
                key,_ = self.keyFc.buildGraph(input, isTraining)      # Shape: [batch*seqLen, outSize]
                value,_ = self.valueFc.buildGraph(input, isTraining)  # Shape: [batch*seqLen, outSize]

                query = permute(query, self.numHeads, seqLen, headSize) # Shape: [batch, numHeads, seqLen, headSize]
                key   = permute(key, self.numHeads, seqLen, headSize)   # Shape: [batch, numHeads, seqLen, headSize]
                value = permute(value, self.numHeads, seqLen, headSize) # Shape: [batch, numHeads, seqLen, headSize]

                # Take the dot product between "query" and "key" to get the raw attention scores.
                attention = tf.matmul( query, key, transpose_b=True)    # Shape: [batch, numHeads, seqLen, seqLen]
                # Scale the attention
                dk = tf.cast(headSize, tf.float32)
                attention = attention / tf.math.sqrt(dk)

                # Apply attention masks (See EmbeddingInLayer::makePlaceholders)
                if self.layers.attentionMasks is not None:
                    attention += self.layers.attentionMasks
                
                attentionProbs = tf.nn.softmax(attention, axis=-1)

                if isTraining:
                    dropOutput = self.buildDropout(attentionProbs, self.dropRate)
                    if dropOutput is not None:      attentionProbs = dropOutput
                
                context = tf.matmul(attentionProbs, value)                  # Shape: [batch, numHeads, seqLen, headSize]
                context = tf.transpose(context, perm=[0, 2, 1, 3])          # Shape: [batch, seqLen, numHeads, headSize]
                context = tf.reshape(context, (-1, self.outSize))           # Shape: [batch*seqLen, outSize]
                                
                outputs += [ attentionProbs ]
                outputs += [ context ]
                
                selfOut,_ = self.selfFc.buildGraph(context, isTraining)
                selfOut,_ = self.selfLn.buildGraph(selfOut+input, isTraining)   # Shape: [batch*seqLen, outSize]
                outputs += [ selfOut ]

            with tf.name_scope('Intermediate'):
                intermediate,_ = self.intermediateFc.buildGraph(selfOut, isTraining)
                outputs += [ intermediate ]

            with tf.name_scope('Output'):
                bertOut,_ = self.outFc.buildGraph(intermediate, isTraining)
                bertOut,_ = self.outLn.buildGraph(bertOut+selfOut, isTraining)
                bertOut = tf.reshape(bertOut, [-1, seqLen, self.outSize])
                outputs += [ bertOut ]

        # The only post activation for BERT is Pooling in the last BERT layer
        self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        headSize = self.outSize//self.numHeads

        s = self.scope + '/'
        onnxBuilder.makeShape(s+'inShape', '-1,outSize')
        onnxBuilder.addNode('Reshape', [inputName, s+'inShape'], [s+'flatIn'], s+'Reshape1')
        queryName = self.queryFc.buildOnnx(onnxBuilder, s+'flatIn')
        keyName = self.keyFc.buildOnnx(onnxBuilder, s+'flatIn')
        valueName = self.valueFc.buildOnnx(onnxBuilder, s+'flatIn')

        onnxBuilder.makeShape(s+'shape01', 'batchSize,seqLen')
        onnxBuilder.addParam(s+'shape23', 'int64', [2], [self.numHeads,headSize])
        onnxBuilder.addNode('Concat', [s+'shape01', s+'shape23'], [s+'intShape'], s+'Concat', axis=0),
        
        def permute(inName, name):
            # batchSize*seqLen,outSize  =>  batchSize,seqLen,numHeads,headSize  =>  batchSize,numHeads,seqLen,headSize
            onnxBuilder.addNode('Reshape', [inName, s+'intShape'], [name+'4D'], name+'/Reshape')
            onnxBuilder.addNode('Transpose', [name+'4D'], [name+'4DT'], name+'/Transpose', perm=[0, 2, 1, 3])
            return name+'4DT'
            
        queryName = permute(queryName, s+'Query')
        keyName = permute(keyName, s+'Key')
        valueName = permute(valueName, s+'Value')

        # Take the dot product between "query" and "key" to get the raw attention scores.
        onnxBuilder.addNode('Transpose', [keyName], [keyName+'T'], s+'Transpose1', perm=[0, 1, 3, 2])
        onnxBuilder.addNode('MatMul', [queryName, keyName+'T'], [s+'AttnRaw'], s+'MatMul1') # batchSize,numHeads,seqLen,seqLen

        onnxBuilder.addParam(s+'dk', 'float', [], [np.sqrt(headSize)])
        onnxBuilder.addNode('Div', [s+'AttnRaw', s+'dk'], [s+'AttnScaled'], s+'Div')

        onnxBuilder.addNode('Add', [s+'AttnScaled', 'AttMask'], [s+'AttnMasked'], s+'Add1')
        onnxBuilder.addNode('Softmax', [s+'AttnMasked'], [s+'AttnProbs'], s+'Softmax', axis=-1)
        
        onnxBuilder.addNode('MatMul', [s+'AttnProbs', valueName], [s+'Context4D'], s+'MatMul2')
        
        # batchSize,numHeads,seqLen,headSize  =>  batchSize,seqLen,numHeads,headSize  =>  batchSize*seqLen,outSize
        onnxBuilder.addNode('Transpose', [s+'Context4D'], [s+'Context4DT'], s+'Transpose2', perm=[0, 2, 1, 3])
        onnxBuilder.addNode('Reshape', [s+'Context4DT', s+'inShape'], [s+'Context'], s+'Reshape2') # batchSize*seqLen,outSize

        selfOutName = self.selfFc.buildOnnx(onnxBuilder, s+'Context')
        onnxBuilder.addNode('Add', [selfOutName, s+'flatIn'], [s+'SelfOut'], s+'Add2')
        selfOutName = self.selfLn.buildOnnx(onnxBuilder, s+'SelfOut')

        intermediateName = self.intermediateFc.buildOnnx(onnxBuilder, selfOutName)
        bertOutName = self.outFc.buildOnnx(onnxBuilder, intermediateName)
        onnxBuilder.addNode('Add', [bertOutName, selfOutName], [s+'BertOutInput'], s+'Add3')
        bertOutName = self.outLn.buildOnnx(onnxBuilder, s+'BertOutInput')

        # The only post activation for BERT is Pooling in the last BERT layer
        outputName = self.buildOnnxPostActivation(onnxBuilder, bertOutName)
        
        return outputName

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
    
        # inputShape is: seqLen,1,outSize,1,1
        headSize = self.outSize//self.numHeads
        # Query:
        # Shape transition:
        #  seqLen,1,outSize,1,1 => seqLen,numHeads,headSize,1,1 => seqLen,numHeads,headSize => numHeads,seqLen,headSize
        sq = s + 'Query/'
        queryName = self.queryFc.buildCml(cmlBuilder, inputName)
        cmlBuilder.add_rank_preserving_reshape(sq+'reshape', queryName, queryName+'Seq.1.nHeads.HeadSize.1',
                                               (0, self.numHeads, headSize, -1, 0))
        cmlBuilder.add_squeeze(sq+'squeeze', queryName+'Seq.1.nHeads.HeadSize.1', queryName+'Seq.nHeads.HeadSize',
                               squeeze_all=True)
        cmlBuilder.add_transpose(sq+'transpose', (1,0,2), queryName+'Seq.nHeads.HeadSize',
                                 queryName+'nHeads.Seq.HeadSize')
        queryName += 'nHeads.Seq.HeadSize'      # Shape: numHeads,seqLen,headSize
        
        # Key:
        # Shape transition:
        #  seqLen,1,outSize,1,1 => seqLen,numHeads,headSize,1,1 => seqLen,numHeads,headSize => numHeads,seqLen,headSize
        sk = s + 'Key/'
        keyName = self.keyFc.buildCml(cmlBuilder, inputName)
        cmlBuilder.add_rank_preserving_reshape(sk+'reshape', keyName, keyName+'Seq.1.nHeads.HeadSize.1',
                                               (0, self.numHeads, headSize, -1, 0))
        cmlBuilder.add_squeeze(sk+'squeeze', keyName+'Seq.1.nHeads.HeadSize.1', keyName+'Seq.nHeads.HeadSize',
                               squeeze_all=True)
        cmlBuilder.add_transpose(sk+'transpose', (1,0,2), keyName+'Seq.nHeads.HeadSize', keyName+'nHeads.Seq.HeadSize')
        keyName += 'nHeads.Seq.HeadSize'        # Shape: numHeads,seqLen,headSize

        # Attention:
        # 'Attention_Mask' was initialized EMB
        sa = s + 'Attn/'
        cmlBuilder.add_batched_mat_mul(sa+'batched_mat_mul', [queryName,keyName], s+'AttnRaw', transpose_b=True)
        cmlBuilder.add_scale(sa+'scale', 1./np.sqrt(headSize), None, False, s+'AttnRaw', s+'AttnScaled', shape_scale=[1])
        cmlBuilder.add_elementwise(sa+'add', [s+'AttnScaled', 'Attention_Mask'], s+'AttnMasked', "ADD")
        cmlBuilder.add_softmax(sa+'softmax', s+'AttnMasked', s+'AttnProbs')
        attName = s+'AttnProbs'                 # Shape: numHeads,seqLen,seqLen

        # Value:
        sv = s + 'Value/'
        valueName = self.valueFc.buildCml(cmlBuilder, inputName)
        cmlBuilder.add_rank_preserving_reshape(sv+'reshape', valueName, valueName+'Seq.1.nHeads.HeadSize.1',
                                               (0, self.numHeads, headSize, -1, 0))
        cmlBuilder.add_squeeze(sv+'squeeze', valueName+'Seq.1.nHeads.HeadSize.1', valueName+'Seq.nHeads.HeadSize',
                               squeeze_all=True)
        cmlBuilder.add_transpose(sv+'transpose', (1,0,2), valueName+'Seq.nHeads.HeadSize', valueName+'nHeads.Seq.HeadSize')
        valueName += 'nHeads.Seq.HeadSize'      # Shape: numHeads,seqLen,headSize

        # Context:
        sc = s + 'Context/'
        cmlBuilder.add_batched_mat_mul(sc+'mat_mul', [attName,valueName], s+'Context3D')
        cmlBuilder.add_transpose(sc+'transpose', (1,0,2), s+'Context3D', s+'Context3DT')
        cmlBuilder.add_rank_preserving_reshape(sc+'reshape', s+'Context3DT', s+'Context3DTR', (0,1,self.outSize))
        cmlBuilder.add_expand_dims(sc+'expand_dims', s+'Context3DTR', s+'Context', axes=[-1,-2])
        contextName = s+'Context'               # Shape:seqLen,1,outSize,1,1
        
        # Self Output
        selfOutName = self.selfFc.buildCml(cmlBuilder, contextName)    # Shape:seqLen,1,outSize,1,1
        cmlBuilder.add_elementwise(s+'add', [selfOutName, inputName], s+'SelfOut', "ADD")
        selfOutName = self.selfLn.buildCml(cmlBuilder, s+'SelfOut')
                
        intermediateName = self.intermediateFc.buildCml(cmlBuilder, selfOutName)
        bertOutName = self.outFc.buildCml(cmlBuilder, intermediateName)
        cmlBuilder.add_elementwise(s+'transpose', [bertOutName, selfOutName], s+'bertOut+selfOut', "ADD")
        bertOutName = self.outLn.buildCml(cmlBuilder, s+'bertOut+selfOut')
                
        # The only post activation for BERT is Pooling in the last BERT layer
        outputName = self.buildCmlPostActivation(cmlBuilder, bertOutName)
        
        return outputName

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('bertLayer'):
            tfBuilder.defineMethod('bertLayer')
            
            # Since GeLU is used inside Bert (not as activation for the whole layer), we
            # need to make sure it is included in the generated code.
            if not tfBuilder.methodDefined('gelu'):
                tfBuilder.defineMethod('gelu')
                tfBuilder.addMethod(("def gelu(self, x, name):",
                                     "    with tf.name_scope(name):",
                                     "        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))",
                                     "        output = x * cdf",
                                     "    return output",
                                     ""))
            
            def fcShapeStr(rankNo, inSize='hiddenSize', outSize='hiddenSize'):
                return "shape = [%s, ranks[%d], %s] if ranks[%d] else [%s, %s]"%(inSize, rankNo, outSize,
                                                                                 rankNo, inSize, outSize)
                
            tfBuilder.addMethod(("def bertLayer(self, layerIn, hiddenSize, interSize, ranks, numHeads, dropRate, ",
                                 "              epsilon, l2LossFactor, cb=12*[0]):",
                                 "    seqLen = tf.shape(layerIn)[1]",
                                 "    headSize = hiddenSize//numHeads",
                                 "",
                                 "    def permute(x, numHeads, seqLen, headSize):",
                                 "        x = tf.reshape(x, [-1, seqLen, numHeads, headSize])",
                                 "        return tf.transpose(x, perm=[0, 2, 1, 3])",
                                 "",
                                 "    def dropOut(input, dropRate):",
                                 "        if dropRate>0:",
                                 "            try:     return tf.nn.dropout(input, rate=dropRate, seed=SEED, name='Dropout')",
                                 "            except:  return tf.nn.dropout(input, keep_prob=1.0-dropRate, seed=SEED, name='Dropout')",
                                 "        return input",
                                 "",
                                 "    " + tfBuilder.getScopeStr('SelfAttention'),
                                 "        " + tfBuilder.getScopeStr('SelfQuery'),
                                 "            " + fcShapeStr(0),
                                 "            query = self.fcLayer(layerIn, shape, True, l2LossFactor, cb[0:2])",
                                 "        " + tfBuilder.getScopeStr('SelfKey'),
                                 "            " + fcShapeStr(1),
                                 "            key = self.fcLayer(layerIn, shape, True, l2LossFactor, cb[2:4])",
                                 "        " + tfBuilder.getScopeStr('SelfValue'),
                                 "            " + fcShapeStr(2),
                                 "            value = self.fcLayer(layerIn, shape, True, l2LossFactor, cb[4:6])",
                                 "",
                                 "        query = permute(query, numHeads, seqLen, headSize)",
                                 "        key   = permute(key, numHeads, seqLen, headSize)",
                                 "        value = permute(value, numHeads, seqLen, headSize)",
                                 "",
                                 "        attention = tf.matmul( query, key, transpose_b=True)",
                                 "        dk = tf.cast(headSize, tf.float32)",
                                 "        attention = attention / tf.math.sqrt(dk)",
                                 "        if self.attentionMasks is not None:",
                                 "            attention += self.attentionMasks",
                                 "        attentionProbs = tf.nn.softmax(attention, axis=-1)",
                                 "        attentionProbs = dropOut(attentionProbs, dropRate)",
                                 "",
                                 "        context = tf.matmul(attentionProbs, value)",
                                 "        context = tf.transpose(context, perm=[0, 2, 1, 3])",
                                 "        context = tf.reshape(context, (-1, hiddenSize))",
                                 "",
                                 "        " + tfBuilder.getScopeStr('SelfOut'),
                                 "            " + fcShapeStr(3),
                                 "            selfOut = self.fcLayer(context, shape, True, l2LossFactor, cb[6:8])",
                                 "            selfOut = dropOut(selfOut, dropRate)",
                                 "        " + tfBuilder.getScopeStr('SelfNorm'),
                                 "            selfOut = self.lnLayer(selfOut+layerIn, [hiddenSize], epsilon)",
                                 "",
                                 "    " + tfBuilder.getScopeStr('Intermediate'),
                                 "        " + tfBuilder.getScopeStr('Intermediate'),
                                 "            " + fcShapeStr(4,'hiddenSize','interSize'),
                                 "            intermediate = self.fcLayer(selfOut, shape, True, l2LossFactor, cb[8:10])",
                                 "            intermediate = self.gelu(intermediate, name='GELU')",
                                 "",
                                 "    " + tfBuilder.getScopeStr('Output'),
                                 "        " + tfBuilder.getScopeStr('Out'),
                                 "            " + fcShapeStr(5,'interSize','hiddenSize'),
                                 "            bertOut = self.fcLayer(intermediate, shape, True, l2LossFactor, cb[10:])",
                                 "            bertOut = dropOut(bertOut, dropRate)",
                                 "        " + tfBuilder.getScopeStr('OutNorm'),
                                 "            bertOut = self.lnLayer(bertOut+selfOut, [hiddenSize], epsilon)",
                                 "        out = tf.reshape(bertOut, [-1, seqLen, hiddenSize])",
                                 "",
                                 "    return out",
                                 ""))
                                 
        if not self.prevLayer.isInput:      layerIn = 'out'
        elif self.prevLayer.name == 'EMB':  layerIn = 'out'
        else:                               layerIn = 'self.modelInput'
        l2LossFactor = tfBuilder.getL2LossFactor(self)
        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope)))
        tfBuilder.graphIndent += 1

        cb, ranks = [], []
        for layer in [self.queryFc, self.keyFc, self.valueFc, self.selfFc, self.intermediateFc, self.outFc]:
            ranks += [layer.rank]
            if tfBuilder.runQuantized:
                cb += [layer.netParams[0].codebookSize]
                cb += [layer.netParams[1].codebookSize if layer.isDecomposed() else 0]
        if sum(ranks) == 0: ranks = '6*[0]'
        
        dropRate = self.dropRate
        if self.model.dropRate == 1:        dropRate = 0                    # Dropout globally disabled
        elif self.dropRate==1:              dropRate = self.model.dropRate  # Use global rate:
        if dropRate<=0.0 or dropRate>=1.0:  dropRate = 0

        if sum(cb) == 0:
            tfBuilder.addToGraph("out = self.bertLayer(%s, %d, %d, %s, %d, %s, %s, %s)",
                                 (layerIn, self.outSize, self.intermediateSize, str(ranks),
                                  self.numHeads, dropRate, self.epsilon, l2LossFactor))
        else:
            tfBuilder.addToGraph(("out = self.bertLayer(%s, %d, %d, %s, %d, %s, %s, %s,",
                                  "                         %s)"),
                                 (layerIn, self.outSize, self.intermediateSize, str(ranks),
                                  self.numHeads, dropRate, self.epsilon, l2LossFactor, str(cb)))
        
        self.buildTfPostActivation(tfBuilder)
        
        tfBuilder.addToGraph("")
        tfBuilder.graphIndent -= 1

    # ******************************************************************************************************************
    def decompose(self, session, decInfo, decomposeDWCN=True):
        newBertParams = []
        numNewParams = 0
        bertInfoStr = self.scope + '\n'
        
        for layer in [self.queryFc, self.keyFc, self.valueFc, self.selfFc, self.selfLn, self.intermediateFc, self.outFc, self.outLn]:
            if layer.name != 'FC':
                newBertParams += NetParam.toNpValues(layer.netParams, session)
                numNewParams += layer.getNumParams()
                continue
            
            newParams, newLayerStr, layerNumNewParams, infoStr = layer.decompose(session, decInfo, decomposeDWCN)
            bertInfoStr += '    ' + infoStr + '\n'

            if newParams is None:
                newParams = NetParam.toNpValues(layer.netParams, session)
                layerNumNewParams = layer.getNumParams()

            newBertParams += newParams
            numNewParams += layerNumNewParams

        return newBertParams, self.getLayerStr(), numNewParams, bertInfoStr[:-1]

# **********************************************************************************************************************
class AggregateFeatureMaps(Layer):
    argsDic = {
                't': ('type', 'fmt', 'ssd'),
                'c': ('numClasses', 'u', None),     # This number includes the background class
              }
    orderedKeys = 'tc'
    name = 'AFM'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        
        # Make internal layers:
        self.classLayers = []
        self.boxLayers = []
        for f, (pa, _) in enumerate(self.layers.paFms):   # Note: the actual feature maps are not available yet.
            numAnchors = pa.anchors
            self.classLayers += [ self.layers.getLayer(2*f, scope='FM%d/Classes'%(f+1),
                                                       layerStr='CONV_K3_O%d_Ps:None'%( numAnchors*self.numClasses ),
                                                       parent=self) ]

            self.boxLayers += [ self.layers.getLayer(2*f+1, scope='FM%d/Boxes'%(f+1),
                                                     layerStr='CONV_K3_O%d_Ps:None'%( numAnchors*4 ), parent=self) ]

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        # For AFM layers input and output shape does not make sense.
        self.inShape = None
        self.outShape = None
        
        for f, (pa, _) in enumerate(self.layers.paFms):
            self.classLayers[f].getOutShape(pa.fmShape)
            self.boxLayers[f].getOutShape(pa.fmShape)

        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        return '%s, %d Feature Maps'%(self.type.upper(), len(self.layers.paFms))

    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        paramStrs = []
        for f, (pa, _) in enumerate(self.layers.paFms):
            layerParamStrs = self.classLayers[f].getAllParamStrs(includeSizeInfo)
            paramStrs += [ '%s/%s'%(self.scope, ps) for ps in layerParamStrs ]

            layerParamStrs = self.boxLayers[f].getAllParamStrs(includeSizeInfo)
            paramStrs += [ '%s/%s'%(self.scope, ps) for ps in layerParamStrs ]

        return paramStrs

    # ******************************************************************************************************************
    def makeL2Loss(self, factor):
        listOfL2Losses = [ netParam.tfL2Loss() for netParam in self.netParams ]
        self.l2Loss = factor * tf.add_n(listOfL2Losses)

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.netParams = []
        if self.type != 'ssd':
            raise NotImplementedError("%s: Currently only SSD object detection type is supported!"%(self.scope))

        initIndex = 0
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            for f, (pa, _) in enumerate(self.layers.paFms):
                numAnchors = pa.anchors
                with tf.name_scope('FM%d'%(f+1)), tf1.variable_scope('FM%d'%(f+1),reuse=tf1.AUTO_REUSE):
                    layerVars = self.classLayers[f].makeVars(None if initValues is None else initValues[initIndex:])
                    initIndex += len(layerVars)
                    self.netParams += layerVars

                    layerVars = self.boxLayers[f].makeVars(None if initValues is None else initValues[initIndex:])
                    initIndex += len(layerVars)
                    self.netParams += layerVars

        return self.netParams
        
    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        classes = []
        boxes = []

        with tf.name_scope(self.scope):
            totalBoxes = 0
            for f, (pa, fm) in enumerate(self.layers.paFms):
                numAnchors = pa.anchors
                numBoxes = pa.fmShape[0]*pa.fmShape[1]*numAnchors
                fm = checkNumeric(fm, "fm contains NAN/INF!!! (Feature Map %d)"%(f) )
                if pa.norm==2:
                    fm = 20.0*tf.math.l2_normalize(fm, axis=-1, epsilon=1e-12)
                with tf.name_scope('FM%d'%(f+1)):
                    classesOut, moments = self.classLayers[f].buildGraph(fm, isTraining)
                    classesOut = tf.reshape(classesOut, [-1, numBoxes*self.numClasses], name='Reshaped')

                    boxesOut, moments = self.boxLayers[f].buildGraph(fm, isTraining)
                    boxesOut = tf.reshape(boxesOut, [-1, numBoxes*4], name='Reshaped')

                classes += [ classesOut ]
                boxes += [ boxesOut ]
                totalBoxes += numBoxes
                                    
            # For each sample in the batch, "tfClasses" has the probability of each class for each anchor in each
            # feature map. "totalBoxes" sets of probabilities for each on of "self.numClasses" classes.
            tfClasses = tf.reshape( tf.concat(classes, axis=1), [-1, totalBoxes, self.numClasses])
            
            # For each sample in the batch, "tfBoxes" has the predicted center points and box size (cx,cy,w,h)
            # adjustments for ewch anchor in each feature map. "totalBoxes" sets of adjustments for each box.
            tfBoxes = tf.reshape( tf.concat(boxes, axis=1), [-1, totalBoxes, 4] )

            outputs = [(tfClasses, tfBoxes)]

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        classOutNames = []
        boxOutNames = []
        
        s = self.deepScope()
        totalBoxes = 0
        for f, (pa, fm) in enumerate(self.layers.paFms):
            numBoxes = pa.fmShape[0]*pa.fmShape[1]*pa.anchors
            fmInputName = pa.inputName
            if pa.norm==2:
                ns = s + 'l2Norm/'
                onnxBuilder.addParam(ns+'epsilon', 'float', [], [1e-12])
                onnxBuilder.addParam(ns+'20.0', 'float', [], [20.0])
                onnxBuilder.addNode('ReduceSumSquare', [fmInputName], [ns+'sumX2'], ns+'ReduceSumSquare', axes=[1])
                onnxBuilder.addNode('Max', [ns+'sumX2', ns+'epsilon'], [ns+'sumX2Epsilon'], ns+'Max')
                onnxBuilder.addNode('Sqrt', [ns+'sumX2Epsilon'], [ns+'L2'], ns+'Sqrt')
                onnxBuilder.addNode('Mul', [fmInputName, ns+'20.0'], [ns+'20x'], ns+'Mul')
                onnxBuilder.addNode('Div', [ns+'20x', ns+'L2'], [fmInputName+'Normalized'], ns+'Div')
                fmInputName = fmInputName+'Normalized'
            cs = s + 'FM%d/Classes/'%(f+1)
            className = self.classLayers[f].buildOnnx(onnxBuilder, fmInputName)
            onnxBuilder.addNode('Transpose', [className], [cs+'ChLast'], cs+'Transpose', perm=[0,2,3,1])
            onnxBuilder.addReshape(cs+'ChLast', [-1, numBoxes*self.numClasses], cs[:-1])
            classOutNames += [ cs[:-1] ]
            
            bs = s + 'FM%d/Boxes/'%(f+1)
            boxName = self.boxLayers[f].buildOnnx(onnxBuilder, fmInputName)
            onnxBuilder.addNode('Transpose', [boxName], [bs+'ChLast'], bs+'Transpose', perm=[0,2,3,1])
            onnxBuilder.addReshape(bs+'ChLast', [-1, numBoxes*4], bs[:-1])
            boxOutNames += [ bs[:-1] ]
            totalBoxes += numBoxes

        onnxBuilder.addNode('Concat', classOutNames, [s+'Classes'], s+'Concat1', axis=1)
        onnxBuilder.addReshape(s+'Classes', [-1, totalBoxes, self.numClasses], s+'AllClasses')

        onnxBuilder.addNode('Concat', boxOutNames, [s+'Boxes'], s+'Concat2', axis=1)
        onnxBuilder.addReshape(s+'Boxes', [-1, totalBoxes, 4], s+'AllBoxes')

        return (s+'AllClasses', s+'AllBoxes')
        
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        classOutNames = []
        boxOutNames = []
        
        numFMs = len(self.classLayers)
        for f, (pa, fm) in enumerate(self.layers.paFms):
            numBoxes = pa.fmShape[0]*pa.fmShape[1]*pa.anchors
            fmInputName = pa.inputName
            if pa.norm==2:
                cmlBuilder.add_l2_normalize(fmInputName + "L2", fmInputName, fmInputName + "L2", epsilon=1e-12)
                cmlBuilder.add_elementwise(fmInputName + "L2x20", fmInputName + 'L2', fmInputName + "L2x20",
                                           "MULTIPLY", 20.0)
                fmInputName += "L2x20"
        
            inName = self.classLayers[f].buildCml(cmlBuilder, fmInputName) # Shape: (1, pa.anchors*numClasses, pa.fmShape[0], pa.fmShape[1] )

            # We first need to convert this to channel-last:
            # FMxClassesChL = Transpose(FMxReshaped),   Shape: 1 x 1 x pa.fmShape[0] x pa.fmShape[1] x pa.anchors*numClasses
            cmlBuilder.add_permute("fm%dClassesChL"%(f+1), (0,2,3,1), inName, "fm%dClassesChL"%(f+1))  # Change to Channel Last
            # FMxClasses = reshape(FMxClassesChL),      Shape: numBoxes x 1 x numClasses x 1 x 1
            cmlBuilder.add_reshape("fm%dClasses"%(f+1), 'fm%dClassesChL'%(f+1),
                                   "fm%dClasses"%(f+1), (numBoxes, self.numClasses, 1, 1), mode=0)
            classOutNames += [ "fm%dClasses"%(f+1) ]
            
            inName = self.boxLayers[f].buildCml(cmlBuilder, fmInputName)
            # We first need to convert this to channel-last:
            # fmxBoxesChL = Transpose(conv output),     Shape: 1 x 1 x pa.fmShape[0] x pa.fmShape[1] x pa.anchors*4
            cmlBuilder.add_permute("fm%dBoxesChL"%(f+1), (0,2,3,1), inName, "fm%dBoxesChL"%(f+1))  # Change to Channel Last
            # fmxClasses = reshape(fmxClassesChL),      Shape: numBoxes x 1 x 4 x 1 x 1)
            cmlBuilder.add_reshape("fm%dBoxes"%(f+1), 'fm%dBoxesChL'%(f+1), "fm%dBoxes"%(f+1), (numBoxes, 4, 1, 1), mode=0)
            boxOutNames += [ "fm%dBoxes"%(f+1) ]

        # allClasses = Concat( classOutNames ),     Shape: totalAnchors x 1 x numClasses x 1 x 1
        cmlBuilder.add_elementwise("allClasses", classOutNames, "allClasses", "SEQUENCE_CONCAT")
        # allBoxDeltas = Concat( boxOutNames ),     Shape: totalAnchors x 1 x 4 x 1 x 1
        cmlBuilder.add_elementwise("allBoxDeltas", boxOutNames, "allBoxDeltas", "SEQUENCE_CONCAT")

        return ("allClasses", "allBoxDeltas")

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if not tfBuilder.methodDefined('makeAnchorBoxes'):
            tfBuilder.defineMethod('makeAnchorBoxes')
            tfBuilder.addToInit(("self.featureMaps = []",
                                 "self.anchorBoxes = self.makeAnchorBoxes()"))
            tfBuilder.addMethod(("def makeAnchorBoxes(self):",
                                 "    res = %d"%(self.layers.input.outShape[0]),
                                 "    minSizes = np.float32([20, 51, 133, 215, 296, 378, 460, 542])",
                                 "    centerSize = []",
                                 "    a = %s"%str([pa.anchors for pa,_ in self.layers.paFms]),
                                 "    d = %s"%str([pa.fmShape[2] for pa,_ in self.layers.paFms]),
                                 "    r = %s"%str([pa.fmShape[0] for pa,_ in self.layers.paFms]),
                                 "    for f, (numAnchors, inDepth, featureMapRes) in enumerate(zip(a,d,r)):",
                                 "        centers = np.arange(res/(2*featureMapRes), res, res/featureMapRes)",
                                 "        cx, cy = np.meshgrid(centers, centers)",
                                 "        cx = (cx.reshape(-1,1)*np.ones((featureMapRes*featureMapRes, numAnchors))).flatten()",
                                 "        cy = (cy.reshape(-1,1)*np.ones((featureMapRes*featureMapRes, numAnchors))).flatten()",
                                 "        minBoxSize, maxBoxSize = minSizes[f:f+2]",
                                 "        w = [ minBoxSize, np.sqrt(minBoxSize*maxBoxSize) ]",
                                 "        h = [ minBoxSize, np.sqrt(minBoxSize*maxBoxSize) ]",
                                 "        aspectRatios = np.float32([x for x in range(2,(numAnchors//2)+1)])",
                                 "        for ar in aspectRatios:",
                                 "            w += [ minBoxSize*np.sqrt(ar), minBoxSize/np.sqrt(ar) ]",
                                 "            h += [ minBoxSize/np.sqrt(ar), minBoxSize*np.sqrt(ar) ]",
                                 "        w = np.float32(featureMapRes*featureMapRes*w)",
                                 "        h = np.float32(featureMapRes*featureMapRes*h)",
                                 "        centerSize += [ np.stack((cx, cy, w, h), axis=1)/res ]",
                                 "    return np.concatenate(centerSize)",
                                 ""))
        
        cbStr = ""
        shapesStr = ""
        numBoxesStr = ""
        totalBoxes = 0
        for f, (pa, fm) in enumerate(self.layers.paFms):
            numAnchors = pa.anchors
            numBoxes = pa.fmShape[0]*pa.fmShape[1]*numAnchors
            totalBoxes += numBoxes
            
            numBoxesStr += ",%d"%(numBoxes)
            shapesStr += ",[%d,"%(pa.fmShape[-1])
            if self.classLayers[f].isDecomposed():
                cbStr += ",[%d, %d]"%(self.classLayers[f].netParams[0].codebookSize,
                                      self.classLayers[f].netParams[1].codebookSize)
                shapesStr += "%d,"%(self.classLayers[f].rank)
            else:
                cbStr += ",[%d, 0]"%(self.classLayers[f].netParams[0].codebookSize)

            shapesStr += "%d]"%(numAnchors*self.numClasses)
            
            shapesStr += ",[%d,"%(pa.fmShape[-1])
            if self.boxLayers[f].isDecomposed():
                cbStr += ",[%d, %d]"%(self.boxLayers[f].netParams[0].codebookSize,
                                      self.boxLayers[f].netParams[1].codebookSize)
                shapesStr += "%d,"%(self.boxLayers[f].rank)
            else:
                cbStr += ",[%d, 0]"%(self.boxLayers[f].netParams[0].codebookSize)
            shapesStr += "%d]"%(numAnchors*4)

        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope)))
        tfBuilder.graphIndent += 1
        
        if tfBuilder.runQuantized:  tfBuilder.addToGraph( "cb = [" + cbStr[1:] + "]" )
        tfBuilder.addToGraph(("numBoxes = [" + numBoxesStr[1:] + "]",
                              "shapes = [" + shapesStr[1:] + "]",
                              "classes = []",
                              "boxes = []",
                              "for f, featureMap in enumerate(self.featureMaps):",
                              "    if f in %s:"%(str([f for f, (pa, _) in enumerate(self.layers.paFms) if pa.norm==2])),
                              "        featureMap = 20.0*tf.math.l2_normalize(featureMap, axis=-1, epsilon=1e-12)",
                              "    " + tfBuilder.getScopeStr("'FM%d'%(f+1)"),
                              "        " + tfBuilder.getScopeStr("Classes"),
                              "            cl = self.convLayer(featureMap, [3,3]+shapes[2*f], (1,1), 'SAME', (1,1), "
                                                              "True, 0%s)"%(", cb[2*f]" if tfBuilder.runQuantized else ""),
                              "            cl = tf.reshape(cl, [-1, numBoxes[f]*%d], name='Reshaped')"%(self.numClasses),
                              "",
                              "        " + tfBuilder.getScopeStr("Boxes"),
                              "            bx = self.convLayer(featureMap, [3,3]+shapes[2*f+1], (1,1), 'SAME', (1,1), "
                                                              "True, 0%s)"%(", cb[2*f+1]" if tfBuilder.runQuantized else ""),
                              "            bx = tf.reshape(bx, [-1, numBoxes[f]*4], name='Reshaped')",
                              "    classes += [cl]",
                              "    boxes += [bx]",
                              "",
                              "tfClasses = tf.reshape( tf.concat(classes, axis=1), [-1, %d, %d])"%(totalBoxes, self.numClasses),
                              "tfBoxes = tf.reshape( tf.concat(boxes, axis=1), [-1, %d, 4] )"%(totalBoxes),
                              ""))
        tfBuilder.graphIndent -= 1

    # ******************************************************************************************************************
    def decompose(self, session, decInfo, decomposeDWCN=True):
        newAfmParams = []
        numNewParams = 0
        afmInfoStr = self.scope + '\n'
        for f, (pa, fm) in enumerate(self.layers.paFms):
            newParams = None
            if self.classLayers[f].name in ['FC', 'CONV', 'DWCN']:
                newParams, newLayerStr, layerNumNewParams, infoStr = self.classLayers[f].decompose(session, decInfo, decomposeDWCN)
                afmInfoStr += '    ' + infoStr + '\n'

            if newParams is None:
                newParams = NetParam.toNpValues(self.classLayers[f].netParams, session)
                layerNumNewParams = self.classLayers[f].getNumParams()

            newAfmParams += newParams
            numNewParams += layerNumNewParams

            newParams = None
            if self.boxLayers[f].name in ['FC', 'CONV', 'DWCN']:
                newParams, decLayerStr, layerNumNewParams, infoStr = self.boxLayers[f].decompose(session, decInfo, decomposeDWCN)
                afmInfoStr += '    ' + infoStr + '\n'

            if newParams is None:
                newParams = NetParam.toNpValues(self.boxLayers[f].netParams, session)
                layerNumNewParams = self.boxLayers[f].getNumParams()

            newAfmParams += newParams
            numNewParams += layerNumNewParams

        return newAfmParams, self.getLayerStr(), numNewParams, afmInfoStr[:-1]


# **********************************************************************************************************************
# MARK: ------------------------ Output Layers ------------------------
class ClassOutLayer(Layer):
    argsDic = {
                'c': ('numClasses', 'u', None),
              }
    orderedKeys = 'c'
    name = 'CLASS'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isOutput = True

    # ******************************************************************************************************************
    def makePlaceholders(self):
        self.placeholders = (tf1.placeholder( tf.int32, shape=[None], name='LabelIndexes'), ) # Shape: [BatchSize]
        return self.placeholders

    # ******************************************************************************************************************
    def postProcessResults(self, rawResults, returnProbs):
        if self.numClasses==2:                                        # Binary Classification
            # rawResults is a 2D matrix (batchSize x 1) with probability of being in class 1
            if returnProbs:                 return rawResults.reshape(-1)   # Shape: [batchSize] (probabilities/float32)
            return np.int32(np.round(rawResults.reshape(-1)))       # Shape: [batchSize] (classIndexes (0/1), int32)
        
        if returnProbs:                     return rawResults   # Shape: [batchSize, numClasses] (probabilities/float32)
        return np.argmax(rawResults, 1)                         # Shape: [batchSize] (classIndexes, int32)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = inShape
        self.outShape = [self.numClasses]
        return self.outShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return '%d classes'%(self.numClasses)

    # ******************************************************************************************************************
    def getOutputStr(self):
        if self.numClasses == 2:
            return 'Probabilities of samples being in class "1".'
            
        return 'Probability distributions for %d classes.'%(self.numClasses)

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        with tf.name_scope(self.scope):
            if isTraining:
                if self.numClasses == 2:
                    tfLabels = tf.cast( tf.reshape(labels[0],[-1,1]), tf.float32 )
                    if self.model.lossFunction is not None:
                        tfLoss = self.model.lossFunction(self.layers, (input,), (tfLabels,))
                    else:
                        tfLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=input, labels=tfLabels))
                elif self.model.lossFunction is not None:
                    tfLoss = self.model.lossFunction(self.layers, (input,), labels)
                else:
                    tfLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input, labels=labels[0]))
                outputs = [ tfLoss ]
            else:
                if self.numClasses == 2:    probs = tf.nn.sigmoid(input)            # Shape: [batchSize]
                else:                       probs = tf.nn.softmax(input, axis=-1)   # Shape: [batchSize, numClasses]
                outputs = [ probs ]

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        if self.numClasses == 2:
            doc = ("For each input sample, this is the probability of the prediction being in class \"1\". For the " +
                   "binary classification model the probability of class \"0\" is one minus this value.")
            onnxBuilder.addParam('ClassProbs', 'float', [-1,1], paramType='output', docStr=doc)
            onnxBuilder.addNode(s+'Sigmoid', [inputName], ['ClassProbs'], s+'Sigmoid')
            
            onnxBuilder.addNode('Round', ['ClassProbs'], [s+'predictedFloat'], s+'Round')
            onnxBuilder.addNode('Cast', [s+'predictedFloat'], ['PredictedClass'], s+'Cast', to=7) # int64

            doc = "A list of integer values specifying the predicted class (0 or 1) for each input sample."
        else:
            doc = ("For each input sample, this is an array of %d probability values, one for each one of " +
                   "the classes.")%(self.numClasses)
            onnxBuilder.addParam('ClassProbs', 'float', [-1]+[self.numClasses], paramType='output', docStr=doc)
            onnxBuilder.addNode('Softmax', [inputName], ['ClassProbs'], s+'Softmax', axis=-1)
        
            onnxBuilder.addNode('ArgMax', ['ClassProbs'], ['PredictedClass'], s+'ArgMax', axis=-1, keepdims=0)

            doc = ("A list of integer values specifying the predicted class (0 to %d) for each input " +
                   "sample.")%(self.numClasses-1)
        onnxBuilder.addParam('PredictedClass', 'int64', [-1], paramType='output', docStr=doc)

        if onnxBuilder.classNames is not None:
            assert len(onnxBuilder.classNames) == self.numClasses
            allClassesStr = ','.join(onnxBuilder.classNames)
            onnxBuilder.addNode('Constant', [], ['ClassNames'], s+'LabelsConstant', value_string=allClassesStr)
            doc = "The comma-separated list of %d class names."%(self.numClasses)
            onnxBuilder.addParam('ClassNames', 'string', [], paramType='output', docStr=doc)

        return ('ClassProbs','PredictedClass')
        
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        del cmlBuilder.spec.description.output[-1]   # Delete the dummy input
        
        s = self.deepScope()
        if self.numClasses==2:
            cmlBuilder.add_activation(s+'SIGMOID', 'SIGMOID', inputName, '1Prob')
            cmlBuilder.add_scale(s+'scale', -1, 1, True, '1Prob', '0Prob', [1], [1])
            cmlBuilder.add_concat_nd(s+'concat', ['0Prob','1Prob'], 'ClassProbs', 0)
        else:
            cmlBuilder.add_softmax(s+'softmax', inputName, 'ClassProbs')
        
        desc = 'A dictionary containing the probability values for each one of %s classes.'%(self.numClasses)
        cmlBuilder.addOutput('ClassProbs', (self.numClasses,), 'float', desc)

        if cmlBuilder.classNames is None:   classNames = [str(i) for i in range(self.numClasses)]
        else:                               classNames = cmlBuilder.classNames
        cmlBuilder.set_class_labels(classNames, 'PredictedLabel', 'ClassProbs')
        cmlBuilder.spec.description.output[-1].shortDescription = 'The text string indicating the predicted label'

        return ('ClassProbs', 'PredictedLabel')

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        tfBuilder.addToInit("self.labels = tf1.placeholder( tf.int32, shape=[None], name='LabelIndexes')")
        tfBuilder.addToInfer(("return self.session.run(self.inferOut, feedDic)",
                              ""))
        tfBuilder.addToTrain(("feedDic[ self.labels ] = batchLabels",
                              "self.session.run(optimizeOp, feedDic)",
                              ""))

        if self.numClasses==2:
            tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                                  "    if isTraining:",
                                  "        self.logits = out",
                                  "        tfLabels = tf.cast( tf.reshape(self.labels, [-1,1]), tf.float32)",
                                  "        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits("
                                                                                        "logits=out, labels=tfLabels))",
                                  "        self.loss += self.l2Loss",
                                  "    else:",
                                  "        self.inferOut = tf.nn.sigmoid(out)",
                                  ""))
                                
        else:
            tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                                  "    if isTraining:",
                                  "        self.logits = out",
                                  "        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits("
                                                                                    "logits=out, labels=self.labels))",
                                  "        self.loss += self.l2Loss",
                                  "    else:",
                                  "        self.inferOut = tf.nn.softmax(out, axis=-1)",
                                  ""))

# **********************************************************************************************************************
class RegOutLayer(Layer):
    argsDic = {
                's': ('shape', 'u*?', [0]),     # The default is shape = [0] which is used for scaler values as label
              }
    orderedKeys = 's'
    name = 'REG'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isOutput = True
        self.supportsEval = not self.isScaler

    # ******************************************************************************************************************
    @property
    def isScaler(self):
        return (self.shape in ([0],[1]))
        
    # ******************************************************************************************************************
    def makePlaceholders(self):
        if self.shape in ([0], [1]):    labelShape = [None]
        else:                           labelShape = [None] + self.shape    # Shape: [BatchSize,...]
        self.placeholders = (tf1.placeholder( tf.float32, shape=labelShape, name='LabelTensors'), )
        return self.placeholders

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        if not self.isScaler:
            assert inShape == self.shape, "Input shape (%s) does not match the specified shape (%s) for REG layer!"%(str(inShape),str(self.shape))
        
        self.inShape = self.outShape = [x for x in inShape]
        return self.outShape

    # ******************************************************************************************************************
    def getOutputStr(self):
        if self.isScaler:               return 'Predicted scaler values.'
        elif len(self.shape)==1:        return 'Predicted Vectors of length %d'%(self.shape[0])
        elif len(self.shape)==2:        return 'Predicted Matrixes with %d rows and %d columns.'%(self.shape[0], self.shape[1])
        return 'Tensors of shape %s.'%(str(self.shape))
    
    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        evalResults = None
        if self.isScaler:   input = tf.reshape(input,[-1])
            
        with tf.name_scope(self.scope):
            if isTraining:
                if self.model.lossFunction is not None: tfLoss = self.model.lossFunction(self.layers, (input,), labels)
                else:                                   tfLoss = tf.reduce_mean( tf.square(input - labels[0]) )
                outputs = [ tfLoss ]
            else:
                if self.supportsEval and (labels is not None):
                    # For multi-dimensional outputs, we calculate the Sum Squared Error and Sum Absolute Error
                    # for each sample
                    rank = len(self.shape) + 1
                    axes = list(range(1,rank))
                    evalResults = (tf.reduce_sum(tf.square(input - labels[0]), axis=axes),
                                   tf.reduce_sum(tf.abs(input - labels[0]), axis=axes) )
                    outputs = [ input ]
                else:
                    outputs = [ input ]

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], evalResults

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        s = self.deepScope()
        outputName = 'Output'
        # Use the identity operator to set the network's output name to "Output".
        onnxBuilder.addNode('Identity',[inputName], [outputName], s+'Identity')
        if len(self.shape)==3:
            # Convert the "self.shape" to channel first
            onnxBuilder.addParam(outputName, 'float', [-1, self.shape[2], self.shape[0], self.shape[1]],
                                 paramType='output', docStr="The predicted output of the model.")
        else:
            onnxBuilder.addParam(outputName, 'float', [-1]+self.shape, paramType='output',
                                 docStr="The predicted output of the model.")
        return outputName
        
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        del cmlBuilder.spec.description.output[-1]   # Delete the dummy input
        if len(self.shape)==3:
            # Assume it is an image and the "self.shape" to channel first
            cmlBuilder.addOutput('Output', (self.shape[2], self.shape[0], self.shape[1]), 'float',
                                 "The predicted output of the model.")
        else:
            cmlBuilder.addOutput('Output', tuple(self.shape), 'float', "The predicted output of the model.")
        return 'Output'

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        if self.shape in ([0], [1]):    labelShape = str([None])
        else:                           labelShape = str([None] + self.shape)    # Shape: [BatchSize,...]
        tfBuilder.addToInit("self.labels = tf1.placeholder(tf.float32, shape=%s, name='LabelTensors')"%(labelShape))
        tfBuilder.addToInfer(("return self.session.run(self.inferOut, feedDic)", ""))
        tfBuilder.addToTrain(("feedDic[ self.labels ] = batchLabels",
                              "self.session.run(optimizeOp, feedDic)",
                              ""))

        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                              "    out = tf.reshape(out,[-1])" if self.isScaler else "",
                              "    if isTraining:",
                              "        self.loss = tf.reduce_mean( tf.square(out - self.labels) )",
                              "    else:",
                              "        self.inferOut = out",
                              ""))
                                
# **********************************************************************************************************************
class AnswerOutLayer(Layer):
    argsDic = {}
    orderedKeys = ''
    name = 'ANSWER'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isOutput = True
        
    # ******************************************************************************************************************
    def makePlaceholders(self):
        self.placeholders = ( tf1.placeholder( tf.int32, shape=[None], name='StartPos'),    # Shape: [BatchSize]
                              tf1.placeholder( tf.int32, shape=[None], name='EndPos') )     # Shape: [BatchSize]
        return self.placeholders

    # ******************************************************************************************************************
    def postProcessResults(self, rawResults, returnProbs):
        # rawResults (startProbs, endProbs)     Shape of each: [batchSize, seqLen]
        if returnProbs:                     return rawResults
        return (np.argmax(rawResults[0],1),np.argmax(rawResults[1],1))  # Shape: [batchSize] (starts, ends)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        # The input shape must be [-1, 2]:
        assert (len(inShape)==2) and (inShape[0] == -1) and (inShape[1] == 2), \
            "%s: The input shape to ANSWER layer must be (seqLen, 2) but it is %s!"%(self.scope, str(inShape))
        self.inShape = [-1, 2]
        self.outShape = [2, -1]
        return self.outShape

    # ******************************************************************************************************************
    def getOutputStr(self):
        return '2 logit vectors (with length ≤ %d) for start and end indexes of the answer.'%(self.layers.input.maxLen)

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        # input shape: [batchSize, seqLen, 2]
        with tf.name_scope(self.scope):
            logits = tf.transpose(input, [2, 0, 1])         # Shape: [2, batchSize, seqLen]
            startEndLogits = tf.unstack(logits, axis=0)
            startLogits = startEndLogits[0]          # Shape: [batchSize, seqLen]
            endLogits = startEndLogits[1]            # Shape: [batchSize, seqLen]

            if isTraining:
                if self.model.lossFunction is not None:
                    loss = self.model.lossFunction(self.layers, (startLogits, endLogits), labels)
                else:
                    startPosLabels, endPosLabels = labels
                    startLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=startPosLabels,
                                                                                              logits=startLogits))
                    endLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=endPosLabels,
                                                                                              logits=endLogits))
                    loss = (startLoss + endLoss)/2.0
                outputs = [ loss ]

            else:
                # Note we return the logits (not probabilities) for start and end positions of the answer. So,
                # there is no need for a softmax here.
                outputs = [(startLogits, endLogits)]

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        # input shape: [batchSize, seqLen, 2]
        doc = ("For each input sample (question/context pair) this is an array of numbers giving the likelihood of " +
               "each token in the sequence being the %s of the answer. To get the actual %s index (%sTok) first get " +
               "the index of highest value in the array (argmax), then subtract the question offset (which is " +
               "length of question tokens plus 2). The actual answer is then:\n\n" +
               "    answerStr = ' '.join(contextTokens[int(startTok):int(endTok+1)])")
        onnxBuilder.addParam('StartLogits', 'float', [-1,-1], paramType='output', docStr=doc%("start", "start", "start"))
        onnxBuilder.addParam('EndLogits', 'float', [-1,-1], paramType='output', docStr=doc%("end", "end", "end"))

        s = self.deepScope()
        onnxBuilder.addNode('Split', [inputName], [s+'startLogits3D', s+'endLogits3d'], s+'Split', axis=2)
        onnxBuilder.addNode('Squeeze', [s+'startLogits3D'], ['StartLogits'], s+'SqueezeStart')  # [batchSize, seqLen]
        onnxBuilder.addNode('Squeeze', [s+'endLogits3d'], ['EndLogits'], s+'SqueezeEnd')        # [batchSize, seqLen]
        return ('StartLogits','EndLogits')

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        s = self.deepScope()
        cmlBuilder.add_slice(s+'slice1', inputName, s+"StartLogits5D", "channel", start_index=0, end_index=1, stride=1)
        cmlBuilder.add_slice(s+'slice2', inputName, s+"EndLogits5D", "channel", start_index=1, end_index=2, stride=1)
        cmlBuilder.add_reshape_static(s+'reshape(Start)', s+"StartLogits5D", 'StartLogits', (-1,))
        cmlBuilder.add_reshape_static(s+'reshape(End)', s+"EndLogits5D", 'EndLogits', (-1,))

        desc = ("An array of numbers giving the likelihood of each token in the sequence being the %s of the answer. " +
                "To get the actual %s index (%sTok) first get the index of highest value in the array (argmax), then " +
                "subtract the question offset (The number of question tokens plus 2). The actual answer is " +
                "then:\n\n    answerStr = ' '.join(contextTokens[int(startTok):int(endTok+1)])")

        del cmlBuilder.spec.description.output[-1]   # Delete the dummy input
        cmlBuilder.addOutput('StartLogits', (cmlBuilder.maxSeqLen,), 'float', desc%("start", "start", "start"))
        cmlBuilder.addOutput('EndLogits', (cmlBuilder.maxSeqLen,), 'float', desc%("end", "end", "end"))
        
        return ('StartLogits','EndLogits')

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        tfBuilder.addToInit(("self.labels = (tf1.placeholder(tf.int32, shape=[None], name='StartPos'),",
                             "               tf1.placeholder(tf.int32, shape=[None], name='EndPos'))"))
        tfBuilder.addToInfer(("startProbs, endProbs = self.session.run(self.inferOut, feedDic)",
                              "return (np.argmax(startProbs,axis=1), np.argmax(endProbs,axis=1))",
                              ""))
        tfBuilder.addToTrain(("# 'batchLabels' must be a tuple of 2 numpy arrays for 'StartPos', and 'EndPos'",
                              "for i in range(2): feedDic[ self.labels[i] ] = batchLabels[i]",
                              "self.session.run(optimizeOp, feedDic)",
                              ""))

        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                              "    logits = tf.transpose(out, [2, 0, 1])",
                              "    startEndLogits = tf.unstack(logits, axis=0)",
                              "    startLogits = startEndLogits[0]",
                              "    endLogits = startEndLogits[1]",
                              "",
                              "    if isTraining:",
                              "        startPosLabels, endPosLabels = self.labels",
                              "",
                              "        startLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=startPosLabels, "
                                                                                                   "logits=startLogits)",
                              "        endLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=endPosLabels, "
                                                                                                 "logits=endLogits)",
                              "        self.loss = (tf.reduce_mean(startLosses) + tf.reduce_mean(endLosses))/2.0",
                              "        self.loss += self.l2Loss",
                              "    else:",
                              "        startProbs = tf.nn.softmax(startLogits, axis=-1)",
                              "        endProbs = tf.nn.softmax(endLogits, axis=-1)",
                              "        self.inferOut = (startProbs, endProbs)",
                              ""))

# **********************************************************************************************************************
class ObjectOutLayer(Layer):
    argsDic = {}
    orderedKeys = ''
    name = 'OBJECT'
    def __init__(self, layers, layerIndex, scope, argsInfo, actStr, parent):
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isOutput = True
        self.anchorBoxes = None
        
    # ******************************************************************************************************************
    def makePlaceholders(self):
        if self.anchorBoxes is None:    self.makeAnchorBoxes()
        numAnchors = self.anchorBoxes.shape[0]
        self.placeholders = ( tf1.placeholder( tf.int32, shape=(None, numAnchors), name='GroundTruthLabels' ),
                              tf1.placeholder( tf.float32, shape=(None, numAnchors, 4), name='GroundTruthBoxes' ),
                              tf1.placeholder( tf.int32, shape=(None, numAnchors), name='GroundTruthMasks' ))
        return self.placeholders

    # ******************************************************************************************************************
    def makeAnchorBoxes(self):
        res = self.layers.input.outShape[0]  # Get Image Resolution from the input layer
        
        # The minSizes gives the size of smallest box for each feature map. The first one is 4 percent of the "res"
        # and the next ones are evenly distributed between 10 to 90 percent of "res".
        minSizes = np.float32([20, 51, 133, 215, 296, 378, 460, 542])
        centerSize = []
        for f, (pa, _) in enumerate(self.layers.paFms):
            numAnchors = pa.anchors
            inDepth = pa.fmShape[2]  # This is the output depth of the conv layer used as feature map

            featureMapRes = pa.fmShape[0]
            centers = np.arange(res/(2*featureMapRes), res, res/featureMapRes)
            cx, cy = np.meshgrid(centers, centers)
            
            # The following repeat every item in "cx" and "cy", "numAnchors" times. So, the length of
            # "cx" and "cy" become "featureMapRes*featureMapRes*numAnchors"
            cx = (cx.reshape(-1,1)*np.ones((featureMapRes*featureMapRes, numAnchors))).flatten()
            cy = (cy.reshape(-1,1)*np.ones((featureMapRes*featureMapRes, numAnchors))).flatten()

            # Now make box sizes (w,h)
            minBoxSize, maxBoxSize = minSizes[f:f+2]
            w = [ minBoxSize, np.sqrt(minBoxSize*maxBoxSize) ]
            h = [ minBoxSize, np.sqrt(minBoxSize*maxBoxSize) ]
            aspectRatios = np.float32([x for x in range(2,(numAnchors//2)+1)])
            for ar in aspectRatios:
                w += [ minBoxSize*np.sqrt(ar), minBoxSize/np.sqrt(ar) ]
                h += [ minBoxSize/np.sqrt(ar), minBoxSize*np.sqrt(ar) ]

            # "w" and "h" are now arrays of length "numAnchors". We now repeat them "featureMapRes*featureMapRes" times.
            # So, the length of "w" and "h" become "featureMapRes*featureMapRes*numAnchors"
            w = np.float32(featureMapRes*featureMapRes*w)
            h = np.float32(featureMapRes*featureMapRes*h)
            centerSize += [ np.stack((cx, cy, w, h), axis=1)/res ]

        # The final "anchorBoxes" contains centers and sizes for all the boxes.([cx, cy, w, h], Shape: nx4)
        self.anchorBoxes = np.concatenate(centerSize)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.numClasses = self.prevLayer.numClasses # Get it from prev layer (AFM)
        self.inShape = None
        self.outShape = None

        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        return '%d Anchor Boxes'%(self.anchorBoxes.shape[0])

    # ******************************************************************************************************************
    def getOutputStr(self):
        return 'A tuple of class labels, boxes, class probabilities, and number of detections.'
        
    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        classes = []
        boxes = []

        with tf.name_scope(self.scope):
            tfClasses, tfBoxes = input

            if isTraining:
                if self.model.lossFunction is not None:
                    loss = self.model.lossFunction(self.layers, input, labels)
                else:
                    gtBatchLabelsPh, gtBatchBoxAdjPh, gtBatchMaskPh = labels
                    tfGtLabels = tf.reshape(gtBatchLabelsPh, [-1,1])    # Shape: (batchSize*numAnchors,1)
                    tfGtBoxes = tf.reshape(gtBatchBoxAdjPh, [-1,4])     # Shape: (batchSize*numAnchors,4)
                    tfGtMasks = tf.reshape(gtBatchMaskPh, [-1])         # Shape: (batchSize*numAnchors,)
                    
                    tfClasses = tf.reshape(tfClasses, [-1, tf.shape(tfClasses)[2]])     # Shape: (batchSize*numAnchors, 81)
                    tfBoxes = tf.reshape(tfBoxes, [-1, 4])                              # Shape: (batchSize*numAnchors, 4)

                    with tf.name_scope('HardNegativeMining'):
                        # Now doing hard negative mining:
                        # We want to find sets of foreground items and background items to use for loss calculations.
                        fgMask = tf.cast(tf.math.equal(tfGtMasks, 1), tf.float32)   # 1.0 for foreground items 0 otherwise

                        fgIndexes = tf.compat.v2.where(tf.math.equal(tfGtMasks, 1))        # The indexes of foreground items

                        # Get number of background items which is 3 times the number of foreground items (or all background items
                        # if it is less than 3 times foreground).
                        fgCount = tf.cast(tf.reduce_sum(fgMask), tf.int32)  # number of foreground items
                        bgCount = tf.math.minimum( tf.shape(fgMask)[0] - fgCount, fgCount * 3)
                        
                        # Calculate the bgScores:
                        # bgScores is -1 for foreground items and -score for background items.
                        bgScores = tf.nn.softmax(tfClasses)[:,0]        # Scores of background class (0) for all items
                        bgScores = -(bgScores * (1-fgMask) + fgMask)    # scores = -(fgMask + score*bgMask)
                        _, bgIndexes = tf.math.top_k(bgScores, k=bgCount)

                    with tf.name_scope('Loss'):
                        fgGtLabels = tf.gather( tfGtLabels, fgIndexes)
                        bgGtLabels = tf.gather( tfGtLabels, bgIndexes)
                        
                        fgClasses = tf.gather( tfClasses, fgIndexes)
                        bgClasses = tf.gather( tfClasses, bgIndexes)
                        
                        fgLoss = tf.reduce_mean( tf1.losses.sparse_softmax_cross_entropy( logits=fgClasses, labels=fgGtLabels) )
                        bgLoss = tf.reduce_mean( tf1.losses.sparse_softmax_cross_entropy( logits=bgClasses, labels=bgGtLabels) )
                        classesLoss = fgLoss + bgLoss
                        classesLoss = tf.debugging.check_numerics(classesLoss, "classesLoss contains NAN/INF!!!" )

                        fgGtBoxes = tf.gather( tfGtBoxes, fgIndexes)
                        fgBoxes = tf.gather( tfBoxes, fgIndexes)
                        boxesLoss = tf1.losses.huber_loss(fgGtBoxes, fgBoxes, delta=0.5)
                        boxesLoss = tf.debugging.check_numerics(boxesLoss, "boxesLoss contains NAN/INF!!!" )

                        loss = tf.add(classesLoss, boxesLoss, "TotalLoss")
                    outputs = [ loss ]
            else:
                # For Inference mode, we apply the predicted adjustments to the anchors to get the predicted boxes:
                centerVar, sizeVar = 0.1, 0.2   # Variance values
                anchorCenters, anchorSizes = self.anchorBoxes[:,:2], self.anchorBoxes[:,2:]
                numAnchors = len(anchorCenters)
                
                tfAnchorCenters = tf.constant(anchorCenters, tf.float32)
                tfAnchorSizes = tf.constant(anchorSizes, tf.float32)

                tfBoxesCenterSize = tf.reshape(tfBoxes, (-1, 2, 2))
                tfCenterDelta, tfSizeDelta = tf.split(tfBoxesCenterSize, 2, axis=1)
                tfCenterDelta = tf.reshape(tfCenterDelta, (-1, numAnchors, 2))
                tfSizeDelta = tf.reshape(tfSizeDelta, (-1, numAnchors, 2))
                    
                newSizes = tf.exp( tfSizeDelta * sizeVar ) * tfAnchorSizes
                newCenters = tfCenterDelta * centerVar * anchorSizes + tfAnchorCenters

                # Now tfBoxes contains predicted boxes (x1,y1,x2,y2), shape is the same: [batchSize, numAnchors, 4]
                tfBoxes = tf.concat([newCenters-newSizes/2.0, newCenters+newSizes/2.0], axis=-1)

                # Now doing the actual NMS:
                tfScores = tf.nn.softmax(tfClasses)
                tfBoxes = tf.expand_dims(tfBoxes, 2)  # Make the shape: [batch, numBoxes, 1, 4]
                
                # Set background scores to all zeros so it never gets selected in NMS.
                maskBg = np.float32([0] + [1]*(self.numClasses-1))
                tfScores = tfScores * maskBg
                
                # This does NMS per-class, per image, for all classes in all images in a batch.
                # The results is a tuple containing the following:
                #   NmsedBoxes:         a list of boxes for each detection for each image
                #   NmsedScores:        a list of probabilities (Confidence) for each detection for each image
                #   NmsedClasses:       a list of predicted labels for each detection for each image
                #   NumDetections:      number of detections for each image
                config = self.model.config
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression( boxes=tfBoxes,
                                                    scores =                    tfScores,
                                                    max_output_size_per_class = config.maxDetectionPerClass,
                                                    max_total_size =            config.maxDetectionsPerImage,
                                                    iou_threshold =             config.iouThreshold,
                                                    score_threshold =           config.scoreThreshold )
                outputs = [tfScores, tfBoxes, (classes, boxes, scores, valid_detections)]

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], None

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputNames):
        s = self.deepScope()
        allClassesName, allBoxesName = inputNames   # Shapes: (-1,totalBoxes,numClasses) , (-1,totalBoxes,4)

        # For Inference mode, we apply the predicted adjustments to the anchors to get the predicted boxes:
        centerVar, sizeVar = 0.1, 0.2   # Variance values
        anchorCenters, anchorSizes = self.anchorBoxes[:,:2], self.anchorBoxes[:,2:]
        numAnchors = len(anchorCenters)
        
        onnxBuilder.addParam(s+'anchorCenters', 'float', anchorCenters.shape, anchorCenters)
        onnxBuilder.addParam(s+'anchorSizes', 'float', anchorSizes.shape, anchorSizes)

        onnxBuilder.addReshape(allBoxesName, [-1,2,2], s+'BoxesCenterSize')
        onnxBuilder.addNode('Split', [s+'BoxesCenterSize'], [s+'CenterDelta', s+'SizeDelta'], s+'Split1', axis=1)
        onnxBuilder.addReshape(s+'SizeDelta', [-1, numAnchors, 2], s+'SizeDeltaAnchors')
        onnxBuilder.addParam(s+'sizeVar', 'float', [], [sizeVar])
        onnxBuilder.addNode('Mul', [s+'sizeVar',s+'SizeDeltaAnchors'], [s+'SizeDeltaXvar'], s+'Mul1')
        onnxBuilder.addNode('Exp', [s+'SizeDeltaXvar'], [s+'ExpSizeDeltaXvar'], s+'Exp')
        onnxBuilder.addNode('Mul', [s+'anchorSizes',s+'ExpSizeDeltaXvar'], [s+'newSizes'], s+'Mul2')
        
        onnxBuilder.addParam(s+'centerVar', 'float', [], [centerVar])
        onnxBuilder.addNode('Mul', [s+'centerVar',s+'anchorSizes'], [s+'anchorSizesXvar'], s+'Mul3')
        onnxBuilder.addReshape(s+'CenterDelta', [-1, numAnchors, 2], s+'CenterDeltaAnchors')
        onnxBuilder.addNode('Mul', [s+'CenterDeltaAnchors',s+'anchorSizesXvar'], [s+'CenterDeltaXanchorSizesXvar'], s+'Mul4')
        onnxBuilder.addNode('Add', [s+'CenterDeltaXanchorSizesXvar',s+'anchorCenters'], [s+'newCenters'], s+'Add1')

        onnxBuilder.addParam(s+'2', 'float', [], [2.0])
        onnxBuilder.addNode('Div', [s+'newSizes',s+'2'], [s+'newSizes/2'], s+'Div')
        onnxBuilder.addNode('Sub', [s+'newCenters',s+'newSizes/2'], [s+'XY1'], s+'Sub')
        onnxBuilder.addNode('Add', [s+'newCenters',s+'newSizes/2'], [s+'XY2'], s+'Add2')
        onnxBuilder.addNode('Concat', [s+'XY1',s+'XY2'], [s+'NmsBoxes'], s+'Concat1', axis=-1)

        onnxBuilder.addNode('Softmax', [allClassesName], [s+'Scores'], s+'Softmax', axis=-1)
        maskBg = np.float32([0] + [1]*(self.numClasses-1))
        onnxBuilder.addParam(s+'maskBg', 'float', maskBg.shape, maskBg)
        onnxBuilder.addNode('Mul', [s+'Scores',s+'maskBg'], [s+'NmsScores0'], s+'Mul5')

        doc = ("This optional integer value defines the max number of object detected in an image per class, " +
               "used by the Non-Maximum-Suppression algorithm. The default is 20.")
        onnxBuilder.addParam("NmsBoxesPerClass", 'int64', [], [20], paramType='input', docStr=doc)
        
        doc = ("This optional floating point value defines the IOU threshold for the Non-Maximum-Suppression " +
               "algorithm. The default is 0.45.")
        onnxBuilder.addParam("NmsIouThreshold", 'float', [], [0.45], paramType='input', docStr=doc)

        doc = ("This optional floating point value defines the threshold for the score (probability) of the " +
               "detected objects to be considered by the Non-Maximum suppression. The default is 0.50.")
        onnxBuilder.addParam("NmsScoreThreshold", 'float', [], [0.5], paramType='input', docStr=doc)

        # ONNX expects the classes to be in the form of [batchSize, numClasses, totalBoxes]:
        onnxBuilder.addNode('Transpose', [s+'NmsScores0'], [s+'NmsScores'], s+'Transpose', perm=[0,2,1])

        onnxBuilder.addNode('NonMaxSuppression',
                            [s+'NmsBoxes', s+'NmsScores', "NmsBoxesPerClass", "NmsIouThreshold", "NmsScoreThreshold"],
                            [s+'NmsIndexes'], s+'NonMaxSuppression', center_point_box=0)
        
        onnxBuilder.addNode('Split', [s+'NmsIndexes'], [s+'BatchNums', s+'Classes', s+'boxIdxes'], s+'Split2', axis=1)
        onnxBuilder.addNode('Concat', [s+'BatchNums',s+'boxIdxes'], [s+'indexes'], s+'Concat2', axis=1)

        # BatchNums: a list specifying batch numbers (Image Num) for each detection
        doc = ("This is a list of integers where the i'th value gives the index of the input image that contains " +
               "the i'th detected box. If only one image was given to the model for inference, then this list " +
               "would contain all zeros because all detected objects belong to the first (only) image. When more " +
               "than one image are given to the model, this can be used to correlate different detections with " +
               "different images.")
        onnxBuilder.addNode('Squeeze', [s+'BatchNums'], ['BatchNums'], name=s+'Squeeze1')
        onnxBuilder.addParam('BatchNums', 'int64', [-1], paramType='output', docStr=doc)

        # Classes: a list specifying the class for each detection (Background not considered as a class)
        doc = "This is a list of integers where the i'th value gives the predicted class for the i'th detected object."
        onnxBuilder.addParam(s+'1', 'int64', [], [1])
        onnxBuilder.addNode('Sub', [s+'Classes',s+'1'], [s+'ClassesNoBg'], name=s+'Sub1')
        onnxBuilder.addNode('Squeeze', [s+'ClassesNoBg'], ['Classes'], name=s+'Squeeze2')
        onnxBuilder.addParam('Classes', 'int64', [-1], paramType='output', docStr=doc)

        # Scores: a 2-D matrix. Element i,j is the probability of j'th class for i'th detection. (Background not
        # considered as a class)
        doc = ("This is a 2-D matrix whose i'th row gives the probabilities of the i'th detected object " +
               "being in each one of %d classes. The index of the highest value gives the predicted class which is " +
               "also available as the i'th value in the \"Classes\" output.")%(self.numClasses-1)
        onnxBuilder.addNode('GatherND', [s+'NmsScores0',s+'indexes'], [s+'ScoresBg'], s+'GatherND2')
        onnxBuilder.addParam(s+'SliceStarts', 'int64', [2], [0,1])
        onnxBuilder.addParam(s+'SliceEnds', 'int64', [2], [1000000,self.numClasses])
        onnxBuilder.addNode('Slice', [s+'ScoresBg', s+'SliceStarts', s+'SliceEnds'], ['Scores'], s+'Slice')
        onnxBuilder.addParam('Scores', 'float', [-1,-1], paramType='output', docStr=doc)
        
        # Boxes: a 2-D matrix. i'th array (size 4) is the box for the i'th detection.
        doc = ("This is a 2-D matrix whose i'th row gives the coordinates of the bounding box for the i'th " +
               "detected object. The box is in the form [x1,y1,x2,y2] where (x1,y1) and (x2,y2) are the top-left " +
               "and bottom-right corners of the box. These values are normalized between 0 and 1. So, to " +
               "get the values in the image coordinates, they must be multiplied by width (x1,x2) and height " +
               "(y1,y2) of the image.")
        onnxBuilder.addNode('GatherND', [s+'NmsBoxes',s+'indexes'], ['Boxes'], s+'GatherND1')
        onnxBuilder.addParam('Boxes', 'float', [-1,-1], paramType='output', docStr=doc)

        if onnxBuilder.classNames is not None:
            assert len(onnxBuilder.classNames) == (self.numClasses-1) # Background class is not included.
            allClassesStr = ','.join(onnxBuilder.classNames)
            onnxBuilder.addNode('Constant', [], ['ClassNames'], 'LabelsConstant', value_string=allClassesStr)
            doc = "The comma-separated list of %d class names."%(self.numClasses-1)
            onnxBuilder.addParam('ClassNames', 'string', [], paramType='output', docStr=doc)

        return ('BatchNums', 'Classes', 'Scores', 'Boxes')

    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        del cmlBuilder.spec.description.output[-1]   # Delete the dummy input
        cmlBuilder.addOutput('AllScores', (self.anchorBoxes.shape[0], self.numClasses), 'double')
        cmlBuilder.addOutput('AllBoxes', (self.anchorBoxes.shape[0], 4), 'double')

        # The outputs of network from AFM layer:
        #       allClasses:   (Shape: totalAnchors x 1 x numClasses x 1 x 1)
        #       allBoxDeltas: (Shape: totalAnchors x 1 x 4 x 1 x 1)
        # Now we want to convert these to AllScores and AllBoxes that can be fed to NMS
        # Get the anchor box information from the model:
        anchorCenters, anchorSizes = self.anchorBoxes[:,:2], self.anchorBoxes[:,2:]     # Shapes: (TotalAnchors, 2)
        totalAnchors = anchorCenters.shape[0]

        # NOTE: add_softmax only operates on the "channels" axis.
        # allScoresBg = Softmax(allClasses),        Shape: totalAnchors x 1 x numClasses x 1 x 1
        cmlBuilder.add_softmax("allScoresBg", 'allClasses', "allScoresBg")     # All scores (background unmasked)
        
        # Set background scores to all zeros so it never gets selected in NMS.
        maskBg = np.float32([0]+[1]*(self.numClasses-1)).reshape((-1,1,1)) # Shape: (numClasses, 1, 1)
        cmlBuilder.add_load_constant("maskBg", "maskBg", maskBg, maskBg.shape) # Shape: 1 x 1 x numClasses x 1 x 1
        # AllScores = allScoresBg * maskBg,         Shape: totalAnchors x 1 x numClasses x 1 x 1
        cmlBuilder.add_elementwise("allScores5D", ['allScoresBg', 'maskBg'], "allScores5D", "MULTIPLY")
        # Reshaped to: 1 x 1 x totalAnchors x numClasses x 1
#        cmlBuilder.add_reshape("AllScores", 'allScores5D', "AllScores", (1, totalAnchors, self.numClasses, 1), mode=0)
        cmlBuilder.add_reshape_static("AllScores", 'allScores5D', "AllScores", (totalAnchors, self.numClasses))

        # allCenters = slice(allBoxDeltas),         Shape: totalAnchors x 1 x 2 x 1 x 1)
        cmlBuilder.add_slice("allCenters", 'allBoxDeltas', "allCenters", "channel", start_index=0, end_index=2, stride=1)
        
        # AllSizes = slice(allBoxDeltas),           Shape: totalAnchors x 1 x 2 x 1 x 1)
        cmlBuilder.add_slice("allSizes", 'allBoxDeltas', "allSizes", "channel", start_index=2, end_index=-1, stride=1)
                              
        centerVar, sizeVar = 0.1, 0.2   # Variance values
        # SizeFactors = exp(AllSizes*sizeVar),      Shape: totalAnchors x 1 x 2 x 1 x 1)
        cmlBuilder.add_unary("sizeFactors", 'allSizes', "sizeFactors", "exp", scale=sizeVar)
        
        anchorSizes = np.expand_dims(anchorSizes, 2)    # Shape: (totalAnchors, 2, 1), "add_load_constant" needs 3D
        cmlBuilder.add_load_constant("anchorSizes3D", "anchorSizes3D", anchorSizes, shape =anchorSizes.shape)
        # AnchorSizes reshaped to: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_reshape("anchorSizes", 'anchorSizes3D', "anchorSizes", (totalAnchors, 2, 1, 1), mode=0)

        # boxSizes = sizeFactors*anchorSizes = exp(allSizes*sizeVar)*anchorSizes,   # Shape: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_elementwise("boxSizes", ['anchorSizes', 'sizeFactors'], "boxSizes", "MULTIPLY")

        # allCentersSizes = allCenters * boxSizes,          # Shape: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_elementwise("allCentersSizes", ['allCenters', 'boxSizes'], "allCentersSizes", "MULTIPLY")

        # allCentersSizesVar = allCentersSizes * centerVar, # Shape: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_elementwise("allCentersSizesVar", 'allCentersSizes', "allCentersSizesVar", "MULTIPLY", centerVar)

        anchorCenters = np.expand_dims(anchorCenters, 2)    # Shape: (totalAnchors, 2, 1), "add_load_constant" needs 3D
        cmlBuilder.add_load_constant("anchorCenters3D", "anchorCenters3D", anchorCenters, shape = anchorCenters.shape)
        # AnchorSizes reshaped to: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_reshape("anchorCenters", 'anchorCenters3D', "anchorCenters", (totalAnchors, 2, 1, 1), mode=0)

        # boxCenters = allCentersSizesVar + anchorCenters,  # Shape: totalAnchors x 1 x 2 x 1 x 1
        cmlBuilder.add_elementwise("boxCenters", ['allCentersSizesVar', 'anchorCenters'], "boxCenters", "ADD")
        
        # NOTE: Unlike tensorflow, coreML's NMS requires centers points and sizes as input for boxes
        # boxCenterSizes = Concat(boxCenters, boxSizes),    # Shape: totalAnchors x 1 x 4 x 1 x 1
        cmlBuilder.add_elementwise("boxCenterSizes", ['boxCenters', 'boxSizes'], "boxCenterSizes", "CONCAT")
        # Reshaped to: 1 x 1 x totalAnchors x 4 x 1
#        cmlBuilder.add_reshape("AllBoxes", 'boxCenterSizes', "AllBoxes", (1, totalAnchors, 4, 1), mode=0)
        cmlBuilder.add_reshape_static("AllBoxes", 'boxCenterSizes', "AllBoxes", (totalAnchors,4))

        return ("AllScores", "AllBoxes")

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        numAnchors = self.anchorBoxes.shape[0]
        tfBuilder.addToInit(("self.labels = (tf1.placeholder(tf.int32, shape=(None, %d), name='GroundTruthLabels'),",
                             "               tf1.placeholder( tf.float32, shape=(None, %d, 4), name='GroundTruthBoxes'),",
                             "               tf1.placeholder( tf.int32, shape=(None, %d), name='GroundTruthMasks'))"),
                            (numAnchors, numAnchors, numAnchors))
        tfBuilder.addToInit(("self.maxDetectionPerClass = kwargs.get('maxDetectionPerClass', %d)",
                             "self.maxDetectionsPerImage = kwargs.get('maxDetectionsPerImage', %d)",
                             "self.iouThreshold = kwargs.get('iouThreshold', %s)",
                             "self.scoreThreshold = kwargs.get('scoreThreshold', %s)"),
                            (tfBuilder.fbModel.config.maxDetectionPerClass,
                             tfBuilder.fbModel.config.maxDetectionsPerImage,
                             str(tfBuilder.fbModel.config.iouThreshold),
                             str(tfBuilder.fbModel.config.scoreThreshold)))
        
        tfBuilder.addToInfer(("return self.session.run(self.inferOut, feedDic)", ""))
        
        tfBuilder.addToTrain(("# 'batchLabels' must be a tuple of 3 numpy arrays for 'GroundTruthLabels', ",
                              "# 'GroundTruthBoxes', and 'GroundTruthMasks'",
                              "for i in range(3): feedDic[ self.labels[i] ] = batchLabels[i]",
                              "self.session.run(optimizeOp, feedDic)",
                              ""))
                
        tfBuilder.addToGraph((tfBuilder.getScopeStr(self.scope),
                              "    if isTraining:",
                              "        gtBatchLabelsPh, gtBatchBoxAdjPh, gtBatchMaskPh = self.labels",
                              "        tfGtLabels = tf.reshape(gtBatchLabelsPh, [-1,1])",
                              "        tfGtBoxes = tf.reshape(gtBatchBoxAdjPh, [-1,4])",
                              "        tfGtMasks = tf.reshape(gtBatchMaskPh, [-1])",
                              "        tfClasses = tf.reshape(tfClasses, [-1, tf.shape(tfClasses)[2]])",
                              "        tfBoxes = tf.reshape(tfBoxes, [-1, 4])",
                              "        with tf.name_scope('HardNegativeMining'):",
                              "            fgMask = tf.cast(tf.math.equal(tfGtMasks, 1), tf.float32)",
                              "            fgIndexes = tf.compat.v2.where(tf.math.equal(tfGtMasks, 1))",
                              "            fgCount = tf.cast(tf.reduce_sum(fgMask), tf.int32)",
                              "            bgCount = tf.math.minimum( tf.shape(fgMask)[0] - fgCount, fgCount * 3)",
                              "            bgScores = tf.nn.softmax(tfClasses)[:,0]",
                              "            bgScores = -(bgScores * (1-fgMask) + fgMask)",
                              "            _, bgIndexes = tf.math.top_k(bgScores, k=bgCount)",
                              "",
                              "        with tf.name_scope('Loss'):",
                              "            fgGtLabels = tf.gather( tfGtLabels, fgIndexes)",
                              "            bgGtLabels = tf.gather( tfGtLabels, bgIndexes)",
                              "            fgClasses = tf.gather( tfClasses, fgIndexes)",
                              "            bgClasses = tf.gather( tfClasses, bgIndexes)",
                              "            fgLoss = tf.reduce_mean( tf1.losses.sparse_softmax_cross_entropy(logits=fgClasses,"
                                                                                                           "labels=fgGtLabels))",
                              "            bgLoss = tf.reduce_mean( tf1.losses.sparse_softmax_cross_entropy(logits=bgClasses,"
                                                                                                           "labels=bgGtLabels))",
                              "            classesLoss = fgLoss + bgLoss",
                              "            fgGtBoxes = tf.gather( tfGtBoxes, fgIndexes)",
                              "            fgBoxes = tf.gather( tfBoxes, fgIndexes)",
                              "            boxesLoss = tf1.losses.huber_loss(fgGtBoxes, fgBoxes, delta=0.5)",
                              "            self.loss = tf.add(classesLoss, boxesLoss, 'TotalLoss')",
                              "",
                              "    else:",
                              "        centerVar, sizeVar = 0.1, 0.2",
                              "        anchorCenters, anchorSizes = self.anchorBoxes[:,:2], self.anchorBoxes[:,2:]",
                              "        numAnchors = len(anchorCenters)",
                              "        tfAnchorCenters = tf.constant(anchorCenters, tf.float32)",
                              "        tfAnchorSizes = tf.constant(anchorSizes, tf.float32)",
                              "        tfBoxesCenterSize = tf.reshape(tfBoxes, (-1, 2, 2))",
                              "        tfCenterDelta, tfSizeDelta = tf.split(tfBoxesCenterSize, 2, axis=1)",
                              "        tfCenterDelta = tf.reshape(tfCenterDelta, (-1, numAnchors, 2))",
                              "        tfSizeDelta = tf.reshape(tfSizeDelta, (-1, numAnchors, 2))",
                              "        newSizes = tf.exp( tfSizeDelta * sizeVar ) * tfAnchorSizes",
                              "        newCenters = tfCenterDelta * centerVar * anchorSizes + tfAnchorCenters",
                              "        tfBoxes = tf.concat([newCenters-newSizes/2.0, newCenters+newSizes/2.0], axis=-1)",
                              "        tfScores = tf.nn.softmax(tfClasses)",
                              "        tfBoxes = tf.expand_dims(tfBoxes, 2)",
                              "        maskBg = np.float32([0] + [1]*%d)"%(self.numClasses-1),
                              "        tfScores = tfScores * maskBg",
                              "        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(",
                              "                boxes =                     tfBoxes,",
                              "                scores =                    tfScores,",
                              "                max_output_size_per_class = self.maxDetectionPerClass,",
                              "                max_total_size =            self.maxDetectionsPerImage,",
                              "                iou_threshold =             self.iouThreshold,",
                              "                score_threshold =           self.scoreThreshold )",
                              "        self.inferOut = (classes, boxes, scores, valid_detections)",
                              ""))

# **********************************************************************************************************************
# The format is:
#   <LayerTypeName>: (Class, Id)
# The LayerTypeName is lower case and Id must be>0.
Layer.layerClasses = {
    # Input Layers:
    'img': (ImageInLayer, 101),
    'tensor': (TensorInLayer, 102),
    'emb': (EmbeddingInLayer, 103),

    # Hidden Layers:
    'fc': (FcLayer, 1),
    'conv': (ConvLayer, 11),
    'dwcn': (DwConvLayer, 21),
    'bn': (BnLayer, 31),
    'id': (None, 32),
    'ln': (LnLayer, 34),
    'afm': (AggregateFeatureMaps, 41),
    'bert': (BertLayer, 42),

    # Output Layers:
    'class': (ClassOutLayer, 201),
    'reg': (RegOutLayer, 202),
    'object': (ObjectOutLayer, 203),
    'answer': (AnswerOutLayer, 204),
}
    
# **********************************************************************************************************************
# MARK: ------------------------ Post Activations ------------------------
# **********************************************************************************************************************
class PostActivation(object):
    paClasses = {}
    name = "UNKNOWN"
    def __init__(self, layer, argsInfo):
        self.layer = layer
        
        argVals = {argName: argDefault for argName,_,argDefault in self.argsDic.values() }
        self.updateArgVals(argVals, argsInfo)
        self.__dict__.update( argVals )

    # ******************************************************************************************************************
    def __repr__(self):
        """
        __repr__
        Returns a text string briefly describing this instance of "PostActivation".
        """
        retStr = '\n%s Instance:'%(self.__class__.__name__)
        retStr += '\n    %s: %s'%('name', self.name)
        retStr += '\n    %s: %s'%('layer', self.layer.scope)
        for arg in self.__dict__:
            val = self.__dict__[arg]
            if arg in ['name', 'layer']:    continue
            if val is None:                 continue
            retStr += '\n    %s: %s'%(arg, str(val))
        return retStr

    # ******************************************************************************************************************
    @classmethod
    def createInstance(cls, layer, paName, paArgs):
        if paName not in PostActivation.paClasses:
            raise ValueError("%s: Unknown Post Activation type '%s'!"%(layer.scope, paName.upper()))
        paClass = PostActivation.paClasses[paName][0]
        return paClass(layer, paArgs)

    # ******************************************************************************************************************
    def updateArgVals(self, argVals, argsInfo):
        for argInfo in argsInfo:
            if argInfo == '':   continue
            argKey = argInfo[0]
            argValStr = argInfo[1:]
            if argKey not in self.argsDic:
                print("%s-%s: Ignoring unknown field '%s'!"%(self.layer.scope, self.name, argInfo))
                continue
            
            argName, argType, _ = self.argsDic[argKey]
            if argValStr[0] == '%':
                assert self.layer.parent is not None, "%s-%s: % sign is only allowed inside block definition!"%(self.layer.scope,
                                                                                                                self.name)
                argVals[argName] = self.layer.parent.getArgValueByKey(argValStr[1:])
            else:
                argVals[argName] = Layer.getArgValue(argType, argValStr)

        for argName in argVals:
            if argVals[argName] is None:
                raise ValueError("%s-%s: The value of '%s' not specified!"%(self.layer.scope, self.name, argName))

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        return inShape

    # ******************************************************************************************************************
    def getLayerStr(self):
        paStrs = [self.name]
        for argKey in self.argsDic:
            argName, argType, argDefault = self.argsDic[argKey]
            argVal = self.__dict__[argName]
            if argVal == argDefault:    continue
            paStrs += [ '%s%s'%(argKey.upper(), Layer.getArgStr(argVal, argType)) ]
        return '_'.join(paStrs)

    # ******************************************************************************************************************
    def getByteList(self):
        typeId = PostActivation.paClasses[self.name.lower()][1]
        byteList = wl.uint2ByteList(typeId)
        
        # Now add all arguments in the argsDic
        for key in self.orderedKeys:
            (argName, argType, _) = self.argsDic[key]
            argVal = self.__dict__[argName]
            byteList += Layer.getByteListForArg(argType, argVal)
    
        return byteList

# **********************************************************************************************************************
class PaMaxPool(PostActivation):
    argsDic = {
                'k': ('kernel', 'uxu', None),
                's': ('stride', 'uxu', -1),
                'p': ('padding', 'p', 'valid')
              }
    orderedKeys = 'ksp'
    name = 'MP'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if self.stride == -1:   self.stride = self.kernel

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        return applyPadding(inShape, self.kernel, self.stride, self.padding)

    # ******************************************************************************************************************
    def getShortDesc(self):
        detailsStr = 'MP(KSP):%s %s %s'%(Layer.getArgStr(self.kernel, 'uxu'),
                                         Layer.getArgStr(self.stride, 'uxu'),
                                         Layer.getArgStr(self.padding, 'p'))
        return detailsStr

# **********************************************************************************************************************
class PaAvgPool(PostActivation):
    argsDic = {
                'k': ('kernel', 'uxu', None),
                's': ('stride', 'uxu', -1),
                'p': ('padding', 'p', 'valid')
              }
    orderedKeys = 'ksp'
    name = 'AP'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if self.stride == -1:   self.stride = self.kernel

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        return applyPadding(inShape, self.kernel, self.stride, self.padding)

    # ******************************************************************************************************************
    def getShortDesc(self):
        detailsStr = 'AP(KSP):%s %s %s'%(Layer.getArgStr(self.kernel, 'uxu'),
                                         Layer.getArgStr(self.stride, 'uxu'),
                                         Layer.getArgStr(self.padding, 'p'))
        return detailsStr

# **********************************************************************************************************************
class PaGlobalAveragePool(PostActivation):
    argsDic = {}
    orderedKeys = ''
    name = 'GAP'

    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        return [1,1,inShape[2]]

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'Global Avg'

# **********************************************************************************************************************
class PaXformerPool(PostActivation):
    argsDic = {
                'n': ('numVectors', 'u', 1),
              }
    orderedKeys = 'n'
    name = 'TP'

    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        return [inShape[1]*self.numVectors]

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'Pooler' if self.numVectors==1 else 'Pooler (N=%d)'%self.numVectors

# **********************************************************************************************************************
class PaClip(PostActivation):
    argsDic = {
                'h': ('hiVal', 'f', np.inf),
                'l': ('loVal', 'f', -np.inf),
                'n': ('normVal', 'f', np.inf),
              }
    orderedKeys = 'hln'
    name = 'CLP'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if (self.loVal == -np.inf) and (self.hiVal == np.inf) and (self.normVal == np.inf):
            raise ValueError("%s: At least one of Higher or Lower bounds or the Norm value must be specified for 'CLP'!"%(layer.scope))
        if self.loVal >= self.hiVal:
            raise ValueError("%s: The lower bound must be smaller than higher bound for 'CLP'!"%(layer.scope))
        if self.normVal <=0:
            raise ValueError("%s: The Norm Value must be a positive number for 'CLP'!"%(layer.scope))

    # ******************************************************************************************************************
    def getShortDesc(self):
        if self.hiVal == np.inf:  return 'x>' + str(self.loVal)
        if self.loVal == -np.inf: return 'x<' + str(self.hiVal)
        return str(self.loVal) + '<x<' + str(self.hiVal)

# **********************************************************************************************************************
class PaUpSampling(PostActivation):
    argsDic = {
                's': ('scale', 'uxu', None)
              }
    orderedKeys = 's'
    name = 'UP'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        # The following will be set in "getOutShape" function
        self.outX = None
        self.outY = None

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        scaleX, scaleY = self.scale
        self.outX = np.int32(inShape[1]*scaleX)
        self.outY = 1 if inShape[0] == 1 else np.int32(inShape[0]*scaleY)
        return [ self.outY, self.outX, inShape[2] ]

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'UP:%s'%(Layer.getArgStr(self.scale, 'uxu'))

# **********************************************************************************************************************
class PaDropout(PostActivation):
    argsDic = {
                'r': ('dropRate', 'f', 1)
              }
    orderedKeys = 'r'
    name = 'DO'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)

    # ******************************************************************************************************************
    def getShortDesc(self):
        if self.dropRate < 1.0:
            return 'DO:%s'%(Layer.getArgStr(self.dropRate, 'f'))
        return 'DO'

# **********************************************************************************************************************
class PaL2Reg(PostActivation):
    argsDic = {
                'f': ('factor', 'f', 1.0)
              }
    orderedKeys = 'f'
    name = 'L2R'
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'L2'

# **********************************************************************************************************************
class PaFeatureMap(PostActivation):
    argsDic = {
                'a': ('anchors', 'u', None),
                'n': ('norm', 'u', 0),
              }
    orderedKeys = 'an'
    name = 'FM'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        
        self.fmShape = None
        self.fmIndex = len(self.layer.layers.paFms)
        self.layer.layers.paFms += [(self, None)]
        self.inputName = None   # Used for coreML export

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        self.fmShape = [x for x in inShape]
        return inShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'FM%d'%(self.anchors)

# **********************************************************************************************************************
class PaAdd(PostActivation):
    argsDic = {
                'n': ('netmarks', 'u*?', None),
              }
    orderedKeys = 'n'
    name = 'ADD'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if len(self.netmarks)==0:
            raise ValueError("%s: Need at least one layer output for \"ADD\"!"%(layer.scope))

    # ******************************************************************************************************************
    def getOutShape(self, inShape ):
        # The output of all the layers and current layer should have the same shape
        for netmark in self.netmarks:
            if netmark not in self.layer.layers.netmarks:
                raise ValueError("%s: Could not find netmark \"%d\" specified for \"ADD\"!"%(self.layer.scope,
                                                                                             netmark))
            if len(self.netmarks)>1: # In case of only one netmark, allow for broadcasting
                if self.layer.layers.netmarks[ netmark ].outShape != inShape:
                    raise ValueError("%s: All specified netmarks should have the same shape for \"ADD\"!"%(self.layer.scope))

        return inShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'ADD:' + ','.join(str(netmark) for netmark in self.netmarks)

# **********************************************************************************************************************
class PaSelect(PostActivation):
    argsDic = {
                'n': ('netmarks', 'u*?', None),
              }
    orderedKeys = 'n'
    name = 'SEL'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if layer.name != 'FC':
            raise ValueError("%s: \"SEL\" is only supported with \"FC\" layers!"%(layer.scope))
        if len(self.netmarks)<2:
            raise ValueError("%s: Need at least two netmarks for \"SEL\"!"%(layer.scope))

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        for netmark in self.netmarks:
            if netmark not in self.layer.layers.netmarks:
                raise ValueError("%s: Could not find netmark \"%d\" specified for \"SEL\"!"%(self.layer.scope,
                                                                                             netmark))
        if len(self.netmarks)==2:
            if inShape[0] != 1:
                raise ValueError("%s: The output shape of layer must be [1] for 'SEL' but it " \
                                 "is [%d]!"%(self.layer.scope, inShape[0]))
        elif inShape[0] != len(self.netmarks):
            raise ValueError("%s: The output shape of layer must be [%d] for 'SEL' but it "    \
                             "is [%d]!"%(self.layer.scope, len(self.netmarks), inShape[0]))
    
        # The output of all the layers should have the same shape
        return self.layer.layers.netmarks[ self.netmarks[0] ].outShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'SEL:' + ','.join(str(netmark) for netmark in self.netmarks)

# **********************************************************************************************************************
class PaWeightedSum(PostActivation):
    argsDic = {
                'n': ('netmarks', 'u*?', None),
              }
    orderedKeys = 'n'
    name = 'WSUM'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)
        if layer.name != 'FC':
            raise ValueError("%s: \"WSUM\" is only supported with \"FC\" layers!"%(layer.scope))
        if len(self.netmarks)<2:
            raise ValueError("%s: Need at least two netmarks for \"WSUM\"!"%(layer.scope))

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        for netmark in self.netmarks:
            if netmark not in self.layer.layers.netmarks:
                raise ValueError("%s: Could not find netmark \"%d\" specified for \"WSUM\"!"%(self.layer.scope,
                                                                                              netmark))
        if len(self.netmarks)==2:
            if inShape[0] != 1:
                raise ValueError("%s: The output shape of layer must be [1] for 'WSUM' but it " \
                                 "is [%d]!"%(self.layer.scope, inShape[0]))
            if self.layer.activation != 'sig':
                raise ValueError("%s: Sigmoid activation function required for 'WSUM' with 2 " \
                                 "inputs! (%s)"%(self.layer.scope,self.layer.activation))
        elif inShape[0] != len(self.netmarks):
            raise ValueError("%s: The output shape of layer must be [%d] for 'WSUM' but it "    \
                             "is [%d]!"%(self.layer.scope, len(self.netmarks), inShape[0]))
        elif self.layer.activation != 'soft':
            raise ValueError("%s: Softmax activation function required for 'WSUM'!"%(self.layer.scope))
    
        # The output of all the layers should have the same shape
        return self.layer.layers.netmarks[ self.netmarks[0] ].outShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'WSUM:' + ','.join(str(netmark) for netmark in self.netmarks)

# **********************************************************************************************************************
class PaTupple(PostActivation):
    argsDic = {
                'n': ('netmarks', 'u*?', None),
              }
    orderedKeys = 'n'
    name = 'TUP'
    # ******************************************************************************************************************
    def __init__(self, layer, argsInfo):
        super().__init__(layer, argsInfo)

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        for netmark in self.netmarks:
            if netmark not in self.layer.layers.netmarks:
                raise ValueError("%s: Could not find netmark \"%d\" specified for \"TUP\"!"%(self.layer.scope,
                                                                                            netmark))
        return inShape

    # ******************************************************************************************************************
    def getShortDesc(self):
        return 'TUP:' + ','.join(str(netmark) for netmark in self.netmarks)
        
# **********************************************************************************************************************
PostActivation.paClasses = {
    'mp': (PaMaxPool, 1),
    'ap': (PaAvgPool, 2),
    'gap': (PaGlobalAveragePool, 3),
    'up': (PaUpSampling, 4),
    'do': (PaDropout, 5),
    'clp': (PaClip, 6),
    'l2r':  (PaL2Reg, 7),
    'fm':  (PaFeatureMap, 8),
    'tp':  (PaXformerPool, 9),
    'add':  (PaAdd, 10),
    'sel':  (PaSelect, 11),
    'wsum':  (PaWeightedSum, 12),
    'tup':  (PaTupple, 13),
}

# **********************************************************************************************************************
# MARK: ------------------------ Blocks ------------------------
# **********************************************************************************************************************
class BlockInstance(Layer):
    def __init__(self, layers, block, layerIndex, scope, argsInfo, actStr, parent):
        self.argsDic = block.argsDic
        self.orderedKeys = block.orderedKeys
        self.block = block
        self.mergeType = block.mergeType
        self.name = block.name
        super().__init__(layers, layerIndex, scope, argsInfo, actStr, parent)
        self.isBlock = True
        
        self.pathsLayers = []
        for pathIndex in range(len(self.block.pathLayersStrs)):
            self.pathsLayers += [ self.getLayers(pathIndex) ]
    
    # ******************************************************************************************************************
    def getArgValueByKey(self, argValkey):
        argValIndex = None
        if len(argValkey)>1:
            # This is array indexing:
            argValkey, argValIndex = argValkey[0], argValkey[1:]
        
        if argValkey not in self.argsDic:
            raise ValueError("%s: Reference to undefined argument '%s'!"%(self.scope, argValkey))

        argName, argType, _ = self.argsDic[ argValkey ]
        argVal = self.__dict__[argName]
        
        if '*' in argType:
            if argValIndex is None:
                raise ValueError("%s: No index specified referring to array type argument '%s'!"%(self.scope, argName))
            argVal = argVal[ int(argValIndex) ]
        elif argValIndex is not None:
            raise ValueError("%s: Invalid indexing for non-array argument '%s'!"%(self.scope, argName))

        return argVal
    
    # ******************************************************************************************************************
    def getLayers(self, pathIndex):
        layerIndex = 0
        pathLetter = "abcdefghij"[pathIndex]
        layers = []
        layersStr = self.block.pathLayersStrs[ pathIndex ].lower()
        layerStrs = layersStr.split(',')
        for layerStr in layerStrs:
            numSubLayers = 1
            if '*' in layerStr:     numSubLayers, layerStr = layerStr.split('*')
            for i in range(int(numSubLayers)):
                scope = 'B%s%d'%(pathLetter, layerIndex+1)
                layers += [ self.layers.getLayer(layerIndex, scope, layerStr, self.block.refBlocks, self) ]
                layerIndex += 1

        return layers

    # ******************************************************************************************************************
    def getOutShape(self, inShape):
        self.inShape = [x for x in inShape]
        self.outShape = None
        for pathLayers in self.pathsLayers:
            pathOutShape = [x for x in inShape]
            for layer in pathLayers:
                if layer is not None:
                    pathOutShape = layer.getOutShape(pathOutShape)
            if self.outShape is None:
                if self.mergeType == 'pick':  continue # For the pick type, the first path is the selector.
                self.outShape = pathOutShape
            elif self.outShape != pathOutShape:
                raise ValueError("%s: Different output shapes for different paths not supported! (%s vs %s)"%(self.scope,
                                                                                                              str(self.outShape),
                                                                                                              str(pathOutShape)))

        for pa in self.postActivations:
            self.outShape = pa.getOutShape( self.outShape )
        
        return self.outShape
    
    # ******************************************************************************************************************
    def getShortDesc(self):
        totalLayers = 0
        for pathLayers in self.pathsLayers:
            for layer in pathLayers:
                if layer is not None:
                    totalLayers += 1
        if len(self.pathsLayers)==1:    return '%d layers'%(totalLayers)
        return '%d Paths, %d layers'%(len(self.pathsLayers), totalLayers)

    # ******************************************************************************************************************
    def getAllParamStrs(self, includeSizeInfo=False):
        paramStrs = []
        for pathLayers in self.pathsLayers:
            for layer in pathLayers:
                if layer is not None:
                    layerParamStrs = layer.getAllParamStrs(includeSizeInfo)
                    layerParamStrs = [self.scope+'/'+s for s in layerParamStrs]
                    if includeSizeInfo:
                        # Indent the block layers
                        layerParamStrs = ['  '+s for s in layerParamStrs]
                    paramStrs += layerParamStrs
        return paramStrs

    # ******************************************************************************************************************
    def makeVars(self, initValues):
        self.netParams = []
        initIndex = 0
        with tf.name_scope(self.scope), tf1.variable_scope(self.scope,reuse=tf1.AUTO_REUSE):
            for pathLayers in self.pathsLayers:
                for layer in pathLayers:
                    if layer is not None:
                        layerInitValues = None if initValues is None else initValues[initIndex:]
                        layerVars = layer.makeVars(layerInitValues)
                        initIndex += len(layerVars)
                        self.netParams += layerVars

        return self.netParams

    # ******************************************************************************************************************
    def buildGraph(self, input, isTraining, labels=None):
        outputs = []
        moments = []
                        
        with tf.name_scope(self.scope):
            for pathLayers in self.pathsLayers:
                lastLayerOut = input
                for layer in pathLayers:
                    if layer is not None:
                        lastLayerOut, layerMoments = layer.buildGraph(lastLayerOut, isTraining)
                        if layerMoments:    moments += layerMoments

                outputs += [ lastLayerOut ]
        
            if len(self.pathsLayers)>1:
                assert self.mergeType in ['add', 'pick'], "Block \"%s\": Only 'ADD' and 'PICK' merge type supported!"%(self.name)
                if self.mergeType=='add':
                    outputs += [ tf.add_n(outputs, name='BlockMerge') ]
                elif self.mergeType=='pick':
                    assert len(self.pathsLayers)>=3, "Block \"%s\": Need at least 3 paths for a block with merge type 'PICK'!"%(self.name)
                    if len(self.pathsLayers)==3:
                        # There are 2 paths to pick from
                        # The last layer of the picker path (first path) must end with a sigmoid activation
                        # and its shape should be [batchSize, 1]
                        outputs += [ tf.where_v2(tf.less(outputs[0],0.5), outputs[1], outputs[2]) ]
                    else:
                        # The last layer of the picker path (first path) is shape [batchSize, numPaths-1].
                        allPaths = tf.stack(outputs[1:], axis=0)
                        indexes = tf.argmax(outputs[0])
                        outputs += [ tf.gather(allPaths, indexes) ]
        
            self.buildActivation(outputs, isTraining)
            self.buildPostActivation(outputs, isTraining)

        outKey = 'training' if isTraining else 'inference'
        self.layerOuts[outKey] = outputs
        return outputs[-1], moments

    # ******************************************************************************************************************
    def buildOnnx(self, onnxBuilder, inputName):
        pathOutNames = []
        for pathLayers in self.pathsLayers:
            pathOutName = inputName
            for layer in pathLayers:
                if layer is None:           continue
                pathOutName = layer.buildOnnx(onnxBuilder, pathOutName)
            pathOutNames += [ pathOutName ]

        assert len(self.pathsLayers) in [1,2], "Block \"%s\": Exporting to ONNX is not supported for more than 2 paths!"%(self.name)
        if len(self.pathsLayers)==1:
            outputName = pathOutNames[0]
        else:
            outputName = self.scope + '/Merged'
            onnxBuilder.addNode('Add', pathOutNames, [outputName], self.scope + '/Add')

        outputName = self.buildOnnxActivation(onnxBuilder, outputName)
        outputName = self.buildOnnxPostActivation(onnxBuilder, outputName)
        return outputName

    # ******************************************************************************************************************
    def buildTf(self, tfBuilder):
        layerIn = 'self.modelInput' if self.prevLayer.isInput else 'out'
        tfBuilder.addToGraph(("blockInput = %s"%(layerIn),
                              "pathOutputs = []",
                              tfBuilder.getScopeStr(self.scope)))
        tfBuilder.graphIndent += 1
            
        for i,pathLayers in enumerate(self.pathsLayers):
            if i>0:     tfBuilder.addToGraph(("out = blockInput"))
            for layer in pathLayers:
                if layer is not None:
                    layer.buildTf(tfBuilder)
            tfBuilder.addToGraph(("pathOutputs += [ out ]", ""))
            
            if len(self.pathsLayers)>1:
                tfBuilder.addToGraph(("out = tf.add_n(pathOutputs, name='BlockMerge')"))
        
        self.buildTfActivation(tfBuilder)
        self.buildTfPostActivation(tfBuilder)
        tfBuilder.graphIndent -= 1
      
    # ******************************************************************************************************************
    def buildCml(self, cmlBuilder, inputName):
        pathOutNames = []
        s = self.deepScope()
        for pathLayers in self.pathsLayers:
            pathOutName = inputName
            for layer in pathLayers:
                if layer is None:           continue
                pathOutName = layer.buildCml(cmlBuilder, pathOutName)
            pathOutNames += [ pathOutName ]

        if len(self.pathsLayers)>1:
            outputName = s + 'ADD'
            cmlBuilder.add_elementwise(outputName, pathOutNames, outputName, 'ADD')
        else:
            outputName = pathOutNames[0]

        outputName = self.buildCmlActivation(cmlBuilder, outputName)
        return self.buildCmlPostActivation(cmlBuilder, outputName)
    
    # ******************************************************************************************************************
    def createBlockStr(self, newName):
        # Note the block created has no arguments. Everything is fixed!
        blockStrs = [ newName, '', self.mergeType ]
        pathLayersStrs = [ 'ID' if p[0] is None else Layers.getLayersStr(p) for p in self.pathsLayers ]
        blockStrs += [ ';'.join(pathLayersStrs) ]
        return '|'.join(blockStrs)

    # ******************************************************************************************************************
    def decompose(self, session, decInfo, decomposeDWCN=True):
        newBlockStrs = [ '', self.mergeType ]
        newBlockParams = []
        newPathsStrs = []
        blockInfoStr = self.scope + '\n'
        numNewParams = 0
        for pathLayers in self.pathsLayers:
            newPathStrs = []
            for layer in pathLayers:
                if layer is None:
                    newPathStrs += ['ID']
                    continue

                if layer.name not in ['FC', 'CONV', 'DWCN']:
                    newPathStrs += [ layer.getLayerStr() ]
                    layerParams = NetParam.toNpValues(layer.netParams, session)
                    newBlockParams += layerParams
                    numNewParams += sum(x.size for x in layerParams)
                    continue

                newParams, decLayerStr, layerNumNewParams, infoStr = layer.decompose(session, decInfo, decomposeDWCN)
                blockInfoStr += '    ' + infoStr + '\n'
                if newParams is None:
                    newPathStrs += [ layer.getLayerStr() ]
                    layerParams = NetParam.toNpValues(layer.netParams, session)
                    newBlockParams += layerParams
                    numNewParams += sum(x.size for x in layerParams)
                    continue

                newPathStrs += [ decLayerStr ]
                newBlockParams += newParams
                numNewParams += layerNumNewParams

            newPathsStrs += [','.join(newPathStrs)]

        newBlockStrs += [ ';'.join(newPathsStrs) ]
        return newBlockParams, '|'.join(newBlockStrs), numNewParams, blockInfoStr[:-1]

# **********************************************************************************************************************
class Block:
    def __init__(self, blockStr, refBlocks=[]):
        self.name, argsInfo, self.mergeType, paths = blockStr.split('|')
        self.argsDic = {}
        argStrs = argsInfo.split(',')
        self.orderedKeys = ''
        for a, argStr in enumerate(argStrs):
            if argStr =='': continue
            argInfo = argStr.split('_')
            argKey = argInfo[0].lower()
            self.orderedKeys += argKey
            if len(argInfo)==4:     self.argsDic[ argKey ] = tuple([argInfo[1], argInfo[2], Layer.getArgValue(argInfo[2], argInfo[3])])
            elif len(argInfo)==3:   self.argsDic[ argKey ] = tuple(argInfo[1:] + [None])
            else:
                raise ValueError("Block \"%s\" argument %d: At least 3 values 'Letter', 'Name', and 'type' are required!"%(self.name, a+1))
        self.pathLayersStrs = paths.split(';')
        assert len(self.pathLayersStrs)<10, "Block \"%s\": Too many paths specified! (Up to 10 supported)"%(name)
        self.refBlocks = refBlocks
    
    # ******************************************************************************************************************
    def instantiate(self, layers, index, scope, argsInfo, actStr, parent=None):
        return BlockInstance(layers, self, index, scope, argsInfo, actStr, parent)

    # ******************************************************************************************************************
    def getBlockStr(self):
        blockStrs = [self.name]
        argStrs = []
        for argKey in self.argsDic:
            argName, argType, argDefault = self.argsDic[argKey]
            argStr = argKey + '_' + argName + '_' + argType
            if argDefault is not None: argStr += '_' + Layer.getArgStr(argDefault, argType)
            argStrs += [ argStr ]
        blockStrs += [ ','.join(argStrs) ]
        blockStrs += [ self.mergeType ]
        blockStrs += [ ';'.join(self.pathLayersStrs) ]
        return '|'.join(blockStrs)
    
    # ******************************************************************************************************************
    def getByteList(self):
        byteList = wl.str2ByteList(self.name)
        byteList += wl.uint2ByteList(len(self.argsDic))

        for argKey in self.argsDic:
            argName, argType, argDefault = self.argsDic[argKey]
            argStr = argKey + '_' + argName + '_' + argType
            if argDefault is not None: argStr += '_' + Layer.getArgStr(argDefault, argType)
            byteList += wl.str2ByteList(argStr)
        
        mergeTypeIds = { 'add':1 }
        byteList += wl.uint2ByteList(mergeTypeIds[self.mergeType])
        byteList += wl.uint2ByteList(len(self.pathLayersStrs))
        for pathLayersStr in self.pathLayersStrs:
            byteList += wl.str2ByteList(pathLayersStr)
    
        return byteList

    # ******************************************************************************************************************
    def __repr__(self):
        """
        __repr__
        Returns a text string briefly describing this instance of "Block".
        """
        retStr = '\n%s:'%(self.__class__.__name__)
        retStr += '\n    Block Name:  %s'%(self.name)
        if len(self.argsDic)>0:
            retStr += '\n    Arguments (%d):'%(len(self.argsDic))
            retStr += '\n        Key  Name                  Type        Default'
            retStr += '\n        ---  --------------------  ----------  ----------'
            for argKey in self.argsDic:
                argName, argType, argDefault = self.argsDic[argKey]
                retStr +='\n         %-2s  %-20s  %-10s  %-10s'%(argKey, argName, argType,
                                                                 'None' if argDefault is None else '"'+str(argDefault)+'"')
        else:
            retStr += '\n    No Arguments.'

        retStr += '\n    Paths (%d):'%(len(self.pathLayersStrs))
        retStr += '\n        No.  Layers Info Str'
        retStr += '\n        ---  --------------------------------------------------------------------------------'
        for i, pathStr in enumerate(self.pathLayersStrs):
            retStr +='\n         %-2d  %s'%(i+1, pathStr)
        retStr += '\n    Merge Method:  %s\n'%(self.mergeType)
        return retStr

