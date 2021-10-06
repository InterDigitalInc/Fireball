# Copyright (c) 2017-2020 InterDigital AI Research Lab
"""
This file implements the main CoreML export functionality. It uses the "coreMlBuild"
method implemented for each layer in the "layers.py" file to build the whole
CoreML model.

Useful Links:
CoreML Tools Documentation: https://apple.github.io/coremltools/index.html
ML Model Spec: https://apple.github.io/coremltools/coremlspecification/index.html
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 03/22/2020    Shahab Hamidi-Rad       Created the file. First version complete with support for SSD
#                                       object detection networks.
# 08/12/2019    Shahab                  Added support for Input/output layers, moved the functionality
#                                       to each layer's "coreMlBuild" function. Added support for NLP
#                                       BERT models.
# 11/29/2020    Shahab                  Restructured the code. Defined the CmlBuilder class. The "layers.py"
#                                       file was also updated accordingly.
# **********************************************************************************************************************
import numpy as np
import tensorflow as tf
import time

from . import __version__ as fbVersion
from .printutils import myPrint

from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder
from coremltools.models.utils import save_spec
import coremltools
from coremltools.models.pipeline import Pipeline


# **********************************************************************************************************************
def addOutputToSpec(spec, name, shape, typeStr='float', desc=None):
    output = spec.description.output.add()
    output.name = name
    if desc is not None:        output.shortDescription = desc
    if typeStr == 'float':      dtype = coremltools.proto.Model_pb2.ArrayFeatureType.FLOAT32
    elif typeStr == 'double':   dtype = coremltools.proto.Model_pb2.ArrayFeatureType.DOUBLE
    elif typeStr == 'int32':    dtype = coremltools.proto.Model_pb2.ArrayFeatureType.INT32
    elif typeStr == 'string':   dtype = datatypes.String()
    datatypes._set_datatype(output.type, datatypes.Array(*shape), dtype)

# **********************************************************************************************************************
def addInputToSpec(spec, name, shape, typeStr='float', desc=None):
    input = spec.description.input.add()
    input.name = name
    if desc is not None:        input.shortDescription = desc
    if typeStr == 'float':      dtype = coremltools.proto.Model_pb2.ArrayFeatureType.FLOAT32
    elif typeStr == 'double':   dtype = coremltools.proto.Model_pb2.ArrayFeatureType.DOUBLE
    elif typeStr == 'int32':    dtype = coremltools.proto.Model_pb2.ArrayFeatureType.INT32
    elif typeStr == 'string':   dtype = datatypes.String()
    datatypes._set_datatype(input.type, datatypes.Array(*shape), dtype)

# **********************************************************************************************************************
class CmlBuilder(NeuralNetworkBuilder):
    def __init__(self, fbModel):
        self.fbModel = fbModel
        super().__init__( [('dummyIn', datatypes.Array(0))],
                          [('dummyOut', datatypes.Array(0))],
                          'classifier' if self.fbModel.layers.output.name=='CLASS' else None,
                          disable_rank5_shape_mapping=True)

    # ******************************************************************************************************************
    def addOutput(self, name, shape, typeStr='float', desc=None):
        addOutputToSpec(self.spec, name, shape, typeStr, desc)
        
    # ******************************************************************************************************************
    def addInput(self, name, shape, typeStr='float', desc=None):
        addInputToSpec(self.spec, name, shape, typeStr, desc)

    # ******************************************************************************************************************
    def addConv(self, name, inName, outName, inDepth, outDepth, kernelXY, strideXY, dilationXY, padding, w, b):
        # For DWConv:
        #   outDepth = 0
        #   kernel_channels = 1
        #   output_channels = inDepth
        #   groups = inDepth
        isDW = (outDepth==0)
        self.add_convolution(name =              name,
                             kernel_channels =   1 if isDW else inDepth,
                             output_channels =   inDepth if isDW else outDepth,
                             height =            kernelXY[1],
                             width =             kernelXY[0],
                             stride_height =     strideXY[1],
                             stride_width =      strideXY[0],
                             border_mode =       'same' if padding=='same' else 'valid',
                             groups =            inDepth if isDW else 1,
                             W =                 w.getCoreMlWeight(),
                             b =                 None if b is None else b.value(),
                             has_bias =          (b is not None),
                             input_name =        inName,
                             output_name =       outName,
                             dilation_factors =  list(dilationXY),
                             padding_top =       0 if padding in ['valid','same'] else padding[0][0],
                             padding_bottom =    0 if padding in ['valid','same'] else padding[0][1],
                             padding_left =      0 if padding in ['valid','same'] else padding[1][0],
                             padding_right =     0 if padding in ['valid','same'] else padding[1][1],
                             same_padding_asymmetry_mode='BOTTOM_RIGHT_HEAVY',
                             **w.getCoreMlQuantInfo())
         
    # ******************************************************************************************************************
    def export(self, fileName, **kwargs):
        t0 = time.time()
        quiet = kwargs.get("quiet", False)
        self.maxSeqLen = kwargs.get("maxSeqLen", 384)
        self.isBgr = kwargs.get("isBgr", True)
        self.rgbBias = kwargs.get("rgbBias", 0)
        self.scale = kwargs.get("scale", 1.0)
        self.classNames = kwargs.get("classNames", None)
        
        if not quiet:
            myPrint('\nExporting to CoreML model "%s" ... '%(fileName))

        layerInputName = ""
        for l,layer in enumerate(self.fbModel.layers):
            if not quiet:
                myPrint('    Now processing "%s" (layer %d of %d) ... \r'%(layer.scope, l+1, self.fbModel.layers.count), False)
            layerInputName = layer.buildCml(self, layerInputName)
        
        if self.fbModel.layers.output.name == "OBJECT":
            networkModel = coremltools.models.MLModel(self.spec)
            nmsModel = self.getNmsModel()
            coreMlModel = self.makePipeLine(networkModel, nmsModel)
        else:
            coreMlModel = coremltools.models.MLModel(self.spec)

        coreMlModel.author = kwargs.get("author", "Fireball Version " + fbVersion)
        coreMlModel.short_description = kwargs.get("modelDesc", 'Model "%s" exported by Fireball'%(self.fbModel.name))

        if not quiet:
            myPrint('    Exported all %d layers.                               '%(self.fbModel.layers.count))
            myPrint('    Saving to "%s" ... '%(fileName), False)
        coreMlModel.save(fileName)
        if not quiet:   myPrint('Done.\nDone (%.2f Sec.)'%(time.time()-t0))

    # **********************************************************************************************************************
    def getNmsModel(self):
        # Inputs to NMS model are the outputs of the actual network model (allScores", "allBoxes) which are produced
        # by the output layer (See ObjectOutLayer::coreMlBuild)
        totalAnchors = self.fbModel.layers.output.anchorBoxes.shape[0]
        nmsInputFeatures = [('AllScores', datatypes.Array(totalAnchors,self.fbModel.layers.output.numClasses)),
                            ('AllBoxes', datatypes.Array(totalAnchors,4))]

        nmsOutputFeatures = [('scores', datatypes.Array(1)), ('boxes', datatypes.Array(1))]
        
        nms_spec = coremltools.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 3

        ioType = coremltools.proto.Model_pb2.ArrayFeatureType.DOUBLE
        for fName, fType in nmsInputFeatures:
            input = nms_spec.description.input.add()
            input.name = fName
            datatypes._set_datatype(input.type, fType, ioType)
       
        for fName, fType in nmsOutputFeatures:
            output = nms_spec.description.output.add()
            output.name = fName
            datatypes._set_datatype(output.type, fType, ioType)

        outputSizes = [self.fbModel.layers.output.numClasses, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
            del ma_type.shape[:]
            
        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = "AllScores"
        nms.coordinatesInputFeatureName = "AllBoxes"
        nms.confidenceOutputFeatureName = "scores"
        nms.coordinatesOutputFeatureName = "boxes"
        nms.iouThresholdInputFeatureName = "iouThreshold"
        nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

        nms.iouThreshold = 0.6          # This is the default value if "iouThreshold" input is not provided.
        nms.confidenceThreshold = 0.4   # This is the default value if "confidenceThreshold" input is not provided.

        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(self.classNames)

        return coremltools.models.MLModel(nms_spec)

    # **********************************************************************************************************************
    def makePipeLine(self, networkModel, nmsModel):
        pipeline = Pipeline([('dummyIn', datatypes.Array(0))], ['boxes', 'scores'])
        pipeline.add_model(networkModel)
        pipeline.add_model(nmsModel)

        # Copy first input (the image) from the network model
        pipeline.spec.description.input[0].ParseFromString(networkModel._spec.description.input[0].SerializeToString())
        addInputToSpec(pipeline.spec, "IouThreshold", (1,), 'float',
                       "A floating point value that defines the IOU threshold for the Non-Maximum-Suppression "\
                       "algorithm. The default is 0.6.")
        addInputToSpec(pipeline.spec, "ConfidenceThreshold", (1,), 'float',
                       "A floating point value that defines the threshold for the confidence (probability) of the "\
                       "detected objects to be considered by the Non-Maximum suppression. The default is 0.4.")

        pipeline.spec.description.output[0].ParseFromString(nmsModel._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nmsModel._spec.description.output[1].SerializeToString())

        # Add descriptions to the inputs and outputs.
        pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
        pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"
        pipeline.spec.specificationVersion = 3

        return coremltools.models.MLModel(pipeline.spec)

