# Copyright (c) 2019-2020 InterDigital AI Research Lab
"""
An ONNX exporter for Fireball Model. This file defines the OnnxBuilder class which
is used to export Fireball models to ONNX format. The functionality in this file
was tested with ONNX version 1.7.0 and onnxruntime version 1.5.2.

Notes:
- Fireball does not use/support ONNX-ML.
- Note that ONNX uses Channel-First format.

Useful ONNX pages:
Python API Overview:        https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
Operator Schemas:           https://github.com/onnx/onnx/blob/master/docs/Operators.md
onnx-tensorflow backend:    https://github.com/onnx/onnx-tensorflow
onnxruntime:                https://github.com/microsoft/onnxruntime

************************************************************************************************************************
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 07/19/2019              Shahab Hamidi-Rad       Created the file.
# 11/13/2020              Shahab Hamidi-Rad       Restructured the code. Defined the OnnxBuilder
#                                                 class.
# **********************************************************************************************************************
import numpy as np
import time

import onnx
from onnx import helper as oh
from onnx import TensorProto

from .printutils import myPrint
from . import __version__ as fbVersion

# **********************************************************************************************************************
class OnnxBuilder:
    def __init__(self, fbModel):
        self.fbModel = fbModel
        self.inputs = []
        self.nodes = []
        self.inits = []
        self.outputs = []
        self.runQuantized = False
        self.classNames = None
        self.hasDimensionsNode = False

    # ******************************************************************************************************************
    def addNode(self, opType, ins, outs, name=None, **kwargs):
        self.nodes += [ oh.make_node(opType, inputs=ins, outputs=outs, name=name, **kwargs) ]
    
    # ******************************************************************************************************************
    def addConv(self, ins, outs, name, kernels=[1,1], strides=[1,1], dilations=[1,1], padding='valid', group=1):
        if (padding == 'same') or (padding == 'valid'):
            self.addNode('Conv', ins, outs, name, kernel_shape=kernels, strides=strides, dilations=dilations,
                         auto_pad={'same':"SAME_UPPER", 'valid':"VALID"}[padding], group=group)
        else:
            # padding is: [ [top, bottom], [left, right] ]
            # onnx padding must be: [ top, left, bottom, right ]
            onnxPaddingValues = [ padding[0][0], padding[1][0], padding[0][1], padding[1][1] ]
            self.addNode('Conv', ins, outs, name, kernel_shape=kernels, strides=strides, dilations=dilations,
                         auto_pad='NOTSET', pads=onnxPaddingValues, group=group)

    # ******************************************************************************************************************
    def addParam(self, name, typeStr, shape, initVal=None, paramType="constant", docStr=""):
        if typeStr.lower()=='float':    onnxType = TensorProto.FLOAT
        elif typeStr.lower()=='string': onnxType = TensorProto.STRING
        elif typeStr.lower()=='int32':  onnxType = TensorProto.INT32
        elif typeStr.lower()=='int64':  onnxType = TensorProto.INT64
        else:                           raise NotImplementedError("'%s' type not implemented yet!"%(typeStr))

        if paramType.lower()=='output': self.outputs += [ oh.make_tensor_value_info(name, onnxType, shape, docStr) ]
        elif paramType.lower()=='input':self.inputs += [ oh.make_tensor_value_info(name, onnxType,  shape, docStr) ]
        else:                           assert (initVal is not None)
        if initVal is not None:
            if (type(initVal)==np.ndarray) and (len(initVal.shape)>1):
                self.inits += [ oh.make_tensor(name, onnxType, shape, initVal.flatten()) ]
            else:
                self.inits += [ oh.make_tensor(name, onnxType, shape, initVal) ]

    # ******************************************************************************************************************
    def addNetParam(self, name, netParam, onnxShape=None):
        scope = name[:name.rfind('/')]
        onnxParam = netParam.rawVal if netParam.bitMask is None else (netParam.rawVal*netParam.bitMask)
        if onnxShape is None:   onnxShape = netParam.shape

        if len(onnxParam.shape)==4:
            if onnxParam.shape[-1]==1:  onnxParam = np.transpose(onnxParam, (2, 3, 0, 1))   # Depth-wise Conv.
            else:                       onnxParam = np.transpose(onnxParam, (3, 2, 0, 1))   # Regular Conv.
            onnxShape = onnxParam.shape
        
        if (not self.runQuantized) or (netParam.codebook is None):
            if netParam.codebook is not None:   onnxParam = netParam.codebook[onnxParam]
            self.inits += [ oh.make_tensor(name, TensorProto.FLOAT, onnxShape, onnxParam.flatten()) ]
            return
           
        codebookLen = netParam.codebookSize
        onnxType = TensorProto.UINT8 if codebookLen<=256 else TensorProto.UINT16
        self.inits += [ oh.make_tensor(name+'/Indexes', onnxType, onnxShape, onnxParam.flatten()),
                        oh.make_tensor(name+'/Codebook', TensorProto.FLOAT, [codebookLen], netParam.codebook) ]

        self.nodes += [ oh.make_node('Cast', [name+'/Indexes'], [name+'/Indexes32'], name+'/Cast', to=TensorProto.INT32),
                        oh.make_node('Gather', [name+'/Codebook', name+'/Indexes32'], [name], name+'/Gather') ]

    # ******************************************************************************************************************
    def addReshape(self, inputName, shape, outputName=None):
        if outputName is None: outputName = inputName + 'Reshaped'
        shapeName = inputName + '/NewShape'
        if type(shape) == str:  shapeName = self.makeShape(shapeName, shape)
        else:                   self.addParam(shapeName, 'int64', [len(shape)], shape)
        self.addNode('Reshape', [inputName, shapeName], [outputName], inputName+'/Reshape')
        return outputName

    # ******************************************************************************************************************
    def makeShape(self, name, dimsStr):
        nameToIdx = {'batchSize':0, 'seqLen':1, 'outSize':2, 'vocabSize':3, '1':-2, '-1':-1}
        dimStrs = dimsStr.split(',')
        indexes = [nameToIdx[dimStr] for dimStr in dimStrs]
        self.addParam(name+'/Idx', 'int64', [len(indexes)], indexes)
        self.addNode('Gather', ['Dimensions', name+'/Idx'], [name], name+'/Gather')
        return name

    # ******************************************************************************************************************
    def export(self, onnxFilePath, **kwargs):
        quiet = kwargs.get('quiet', False)
        self.runQuantized = kwargs.get('runQuantized', False)
        self.classNames = kwargs.get('classNames', None) # If present, it must be a list of strings one for each class

        t0 = time.time()
        if not quiet:
            myPrint('\nExporting to ONNX model "%s" ... '%(onnxFilePath))

        layerOutputNames = ""
        for l,layer in enumerate(self.fbModel.layers):
            if not quiet:
                myPrint('    Now processing "%s" (layer %d of %d) ...           \r'%(layer.scope, l+1,
                                                                                     self.fbModel.layers.count), False)
            layerOutputNames = layer.buildOnnx(self, layerOutputNames)
        
        if not quiet:
            myPrint('    Processed all %d layers.                                     '%(self.fbModel.layers.count))
            myPrint('    Saving to "%s" ... '%(onnxFilePath), False)

        docString = kwargs.get('graphDocStr', None)
        graph = oh.make_graph(self.nodes, self.fbModel.name, self.inputs, self.outputs, self.inits, docString)
        
        docString = kwargs.get('modelDocStr', "")
        onnxModel = oh.make_model(graph, producer_name='Fireball', producer_version=fbVersion, doc_string=docString,
                                  opset_imports = [oh.make_opsetid("", 17)])
        onnxModel.ir_version = 8
        onnx.save(onnxModel, onnxFilePath)

        if not quiet:
            myPrint('Done.\nDone (%.2f Sec.)'%(time.time()-t0))

