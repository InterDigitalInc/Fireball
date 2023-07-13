*layersInfo*
============
In Fireball the model's network structure is specified by a single text string called *layersInfo*. When a model is defined for the first time, the *layersInfo* should be passed to the :py:class:`fireball.model.Model` class constructor. This information is saved to the fireball model file `(*.fbm)` when the model is saved.

Most common deep neural network structures can be implemented using the Fireball layers. The main building blocks of *layersInfo* are the layers of the network. Several layers can be grouped together. In Fireball these groups are called *stages*.

*layersInfo* Syntax
-------------------
Generally Fireball's *layersInfo* can be represented as a semicolon-separated list of stages::

    LayersInfo: stage1;stage2; ... ;stageN

Each stage contains one or more *layers* which are separated by commas::

    stage: layerInfo1,layerInfo2,...layerInfoM
    
Please note that using *stages* is optional. If there is no semicolon in the *layersInfo*, then all layers are assumed to be in a single (hidden) stage.

Different parts of a *layerInfo* are separated by colons (:). They fall in one of the following three categories:

    * **Layer Type**: The first part of each layer specifies the type of layer (for example a :ref:`Fully Connected layer<FC>` or a :ref:`Convolutional Layer<CONV>`). The related attributes for the layer (such as number of output channels or kernel size) follow the layer type separated by underscore characters. Each attribute is specified by a letter and one or more numbers or letters. Please refer to :ref:`Layer Specifications<LAYERSPECS>` for detail explanation of supported layers. Here are some examples::
    
            # A Fully Connected layer with 128 output channels
                "FC_O128"

            # A convolutional layer with a 3x3 kernel,
            # stride 2, and 16 output channels
                "CONV_K3_S2_O16"

    * **Activation Function**: The second part of a *layerInfo* specifies the activation function. Please refer to the :ref:`Activation Functions<ACTIVATION>` section for a list of supported Activation Functions. Here are some examples::

            # A Fully Connected layer with 128 output channels with
            # Tangent Hyperbolic activation function
                "FC_O128:Tanh"

            # A Convolutional layer, 3x3 kernel, stride 2, 16 output channels,
            # with ReLU activation function
                "CONV_K3_S2_O16:ReLU"


    * **Post-Activations**: In Fireball, any process that occurs at the output of a layer after the activation function and before the next layer, is called *Post-Activation* (for example the average pooling process). Generally post-activation processes do not involve any network parameters. The attributes of a post-activation (such as the drop-rate for a drop-out post-activation) follow the type separated by underscore characters. A layer can have multiple *Post-Activations* that are separated by the colon (:) characters. Please refer to the :ref:`Post-Activations<POSTACTIVATION>` section for a list of supported Post-Activations and their detail explanation. Here are some examples::

            # A Fully Connected layer with 128 output channels with
            # Tangent Hyperbolic activation function, and drop-out with drop-rate 0.3
                "FC_O128:Tanh:DO_R0.3"

            # A Convolutional layer, 3x3 kernel, stride 2, 16 output
            # channels, ReLU activation function, Max Pooling with
            # Kernel and stride 2
                "CONV_K3_S2_O16:ReLU:MP_K2_S2"

Please note that only the layer type is the required part. The activation and post-activations are optional. The following example shows how a simple LeNet-5 model can be implemented in Fireball::

        layersInfo = "IMG_S28_D1,CONV_K5_O6_Ps:Tanh:MP_K2,CONV_K5_O16:Tanh:MP_K2, \
                      FC_O120:Tanh,FC:O84:Tanh,FC_O10,CLASS_C10"
                      
Also note that each *layerInfo* can be prefixed or postfixed with netmark notations. See :ref:`Netmarks<NETMARK>` for more information.

Layer Types
-----------

The following layers are supported by Fireball:
    * Input Layers:
    
        * :ref:`IMG: Image Input <IMG>`
        * :ref:`TENSOR: Tensor Input <TENSOR>`
        * :ref:`EMB: Embedding Input  <EMB>` (used in NLP models)
    
    * Hidden Layers:
    
        * :ref:`FC: Fully Connected <FC>`
        * :ref:`CONV: Convolutional <CONV>`
        * :ref:`DWCN: Depth-wise Convolution <DWCN>`
        * :ref:`BN: Batch Normalization <BN>`
        * :ref:`LN: Layer Normalization <LN>`
        * :ref:`AFM: Aggregate Feature Maps  <AFM>` (used in Object Detection models)
        * :ref:`BERT: Bidirectional Encoder Representations from Transformers  <BERT>` (used in NLP models)
        
    * Output Layers:
    
        * :ref:`CLASS: Classification Output <CLASS>`
        * :ref:`REG: Regression Output <REG>`
        * :ref:`OBJECT: Object Detection Output <OBJECT>`
        * :ref:`ANSWER: Answer Output <ANSWER>` (NLP/Question-answering tasks)
        
Each layer can have zero or more attributes. Each layer attribute is specified by a letter which specifies the attribute (such as 'K' for "Kernel") followed by the value for the attribute which can be one or more numbers or letters.

Here is a list of general rules for the layers:

    * The first layer of the model **MUST** be an input layer.
    * The last layer of the model **MUST** be an output layer.
    * Layer names are case insensitive.
    * Layer attributes are separated by underscore characters.
    * Layer attributes can come in any order.

.. _ACTIVATION:

Activation Functions
--------------------
Currently the following activation functions are supported by Fireball:

    * ReLU: Rectified Linear Unit
    * LReLU: Leaky Rectified Linear Unit
    * GeLU: Gaussian Error Linear Unit
    * SeLU: Scaled Exponential Linear Unit
    * Tanh: Tangent Hyperbolic
    * Sig: Sigmoid
    * Soft: Softmax
    * None: No Activation (Default)
 
Here is a list of general rules for the activation functions:

    * Activation Functions are case insensitive
    * If an Activation Functions is missing the default is "None".

.. _POSTACTIVATION:

Post-Activations
----------------
Currently the following post-activations are supported by Fireball:

    * :ref:`RS: Reshape <RS>`
    * :ref:`MP: Max Pooling <MP>`
    * :ref:`AP: Average Pooling <AP>`
    * :ref:`TP: Transformer pooling <TP>` (NLP)
    * :ref:`GAP: Global Average Pooling <GAP>`
    * :ref:`UP: Upsampling <UP>`
    * :ref:`DO: Dropout <DO>`
    * :ref:`CLP: Clip <CLP>`
    * :ref:`L2R: L2 Regularization <L2R>`
    * :ref:`FM: Feature Map <FM>` (Object Detection)
    * :ref:`ADD: Add netmarks <ADD>`
    * :ref:`SEL: Select netmark <SEL>`
    * :ref:`WSUM: Weighted Sum Netmarks <WSUM>`
    * :ref:`TUP: Tuple netmarks <TUP>`
    * :ref:`RND: Round <RND>`

Each post-activation can have zero or more attributes. Each post-activation attribute is specified by a letter indicating the attribute (For example 'R' for "Drop-Rate") followed by the value for the attribute which can be one or more numbers or letters.

Here is a list of general rules for post-activations:

    * Post-activation names are case insensitive.
    * Post-activation attributes are separated by underscore characters.
    * Post-activation attributes can come in any order.
    * A layer can have zero or more post-activations.
    * In the *layerInfo* string, post-activation should always be after the activation function. If no activation function is used for a layer the ``None`` can be used in its place or it can be left empty. Here are some examples::
    
            # A Fully Connected layer with 128 output channels with
            # no activation function, and drop-out with drop-rate 0.3
                "FC_O128:None:DO_R0.3"   # OK
            # or
                "FC_O128::DO_R0.3"       # OK
            # But this is incorrect:
                "FC_O128:DO_R0.3"        # NOT OK

.. _LAYERSPECS:

Layer Specifications
====================

.. _IMG:

IMG: Image Input Layer
----------------------
Image Input layer is used to feed a Fireball model with Images.

Attributes
^^^^^^^^^^
    size : S
        The image dimensions in the form of `width x height`. For example ``S800x600`` means the model accepts images of the width 800 and height 600. This means the actual tensor shape is (600,800). If height is missing, it is assumed to be the same as width. For example ``S224`` means the model accepts 224x224 square images.

    depth : D, optional, default: 3
        The number of channels for the image. The default is 3 (for RGB or BGR images). For monochrome images use ``D1``.

Here are some examples::

    # An Image Input layer for 800x600 RGB images
        "IMG_S800x600_D3"

    # An Image Input layer for 28x28 monochrome images
        "IMG_S28_D1"

.. _TENSOR:

TENSOR: Tensor Input Layer
--------------------------
Tensor Input layer is used to feed a Fireball model with tensors of the specified shape.

Attributes
^^^^^^^^^^
    shape : S
        The tensor shape. It is a list of positive integers separated by the '/'.
    
Here are a couple of examples::
    
    # A Tensor Input layer for vectors of length 10
        "TENSOR_S10"

    # A Tensor Input layer for matrixes with shape (3,5)
        "TENSOR_S3/5"

.. _EMB:

EMB: Embedding Input Layer
--------------------------
Embedding Input layer is used with NLP and some other Time-Series tasks. Usually the inputs to this layer are prepared by a tokenizer. This layer is designed to work with Fireball's :py:mod:`SQuAD <fireball.datasets.squad>` and :py:mod:`GLUE <fireball.datasets.glue>` datasets and the Tokenizer Class.

For each sample, the embedding input layer receives 2 arrays:

    * TokenIds: A list of integer values that are the token IDs of the tokens in a sequence. The token IDs are actually the indexes to a vocabulary of tokens (Using *WordPiece* subword segmentation algorithm).
    * TokenTypes: A list of integer values with the same length as `TokenIds` that indicate the type of each token in the `TokenIds` list. For example in question-answering tasks the question and context tokens are concatenated and fed to the model as "TokenIds". The `TokenTypes` array has 0's for the question tokens and 1's for the context tokens.
    
The input to this layer is a tuple of tensors (TokenIds, TokenTypes). Each tensor is of the shape (batchSize, maxLen). When a batch of sequences is given to the model (For example during training), the sequences are padded with 0's so that all of them have the same length.

Attributes
^^^^^^^^^^
    outSize : O
        The output size of the embedding layer. This is also known as "Hidden Size".
        
    initStd : S, optional, default: 0.02
        This is the Standard Deviation of the distribution used for random initialization of weight parameters in this layer.
        
    maxLen : L, optional, default: 512
        This is the maximum sequence length supported by this layer (and the model). In other words this is the maximum number of tokens in the inputs to the layer. The default is 512.

    vocabSize : V, optional, default: 30522
        The size of vocabulary. By default, this is set to 30522, which is the total number of tokens defined in WordPiece.
        
    rank : R, optional, default: 0
        This is used for Low-Rank models. Low-Rank decomposition is an algorithm used by Fireball to reduce the number of parameters of a model. If this layer is a low-rank decomposed layer, the rank attribute is a positive number specifying the rank of decomposed word embedding matrix. Otherwise for regular models, this is set to 0 which is the default. In most cases this should be left unchanged when composing *layersInfo*. The method :py:meth:`~fireball.model.Model.createLrModel` can be used to reduce the number of parameters of the model. When this method is called, it automatically assigns a `rank` value for each decomposed layer.

**Different Types of Sequence Length**

When we are talking about sequence length in different NLP tasks it can apply to one of the following types of sequence length. For a better understanding of how the NLP models work, it is important to know the differences:

    * Model's maxLen: This is fixed for the a model design and used during the training of the model. This is the maximum sequence length that can be handled by the model. This cannot be changed after the training. For example for `BERTbase` model this is set to 512. This is defined by this layer's ``maxLen`` attribute.
    
    * Datasets's maxSeqLen: This is the max sequence length that occurs in a dataset. For example for SQuAD, this is set to 384. This value cannot be more than the Model's ``maxLen``.
    
    * seqLen: This is the sequence length for a single sample processed by the model. It may or may not include padding. For processing just one sample, padding is not needed. To process a batch of samples, we use padding to make them the same length.

    * noPadLen: When padding is used, this is the non-padded sequence length. When padding is not used, this is equal to the seqLen. (When processing only one sample for example)

Here is an example for BERTbase model::

    # An Embedding input layer for BERTbase model
        "EMB_L512_O768"
    
.. _FC:

FC: Fully Connected Layer
-------------------------
Fully connected layer also known as *Dense* layer is used for a linear transformation of the input tensor.

Attributes
^^^^^^^^^^
    outSize : O
        The size of output tensors also known as output channels.
        
    rank : R, optional, default: 0
        This is used for Low-Rank models. Low-Rank decomposition is an algorithm used by Fireball to reduce the number of parameters of a model. If this layer is a low-rank decomposed layer, the rank attribute is a positive number specifying the rank of the weight matrix. Otherwise for regular models, this is set to 0 which is the default. In most cases this should be left unchanged when composing *layersInfo*. The method :py:meth:`~fireball.model.Model.createLrModel` can be used to reduce the number of parameters of the model. When this method is called, it automatically assigns a `rank` value for each decomposed layer.

    hasBias : B, optional, default: 1
        This attribute indicates whether a bias is used for this linear transformation. If this is 1 (the default), a bias vector is used in this layer. Otherwise if ``B0`` is included in this *layerInfo*, it means the bias is not used.

Here are a couple of examples::

    # A fully connected layer with 128 output channels with bias
        "FC_O128"

    # A fully connected layer with 256 output channels with no bias
        "FC_O256_B0"

.. _CONV:

CONV: Convolutional Layer
-------------------------
This layer implements a convolution operation on the input tensor.

Attributes
^^^^^^^^^^
    kernel : K
        The kernel size for this layer. For square kernels, only one integer value is enough to specify the kernel size. For example ``K3`` specifies a 3x3 kernel. For non-square kernels, the width and height of the kernel are included and separated by 'x'. For example ``K3x5`` specifies a 3x5 kernel. Please note that the actual *shape* of kernel is (5,3) in this case. (5 rows, 3 columns)
        
    stride : S, optional, default: 1
        The stride of convolution. If the stride is the same for both dimensions, only one integer value is enough to specify the stride. Otherwise the stride along the width and height are included and separated by 'x'. For example ``S2x1`` specifies a stride of 2 along the width and 1 along height.

    outDept : O
        The output depth of convolution also known as the number of output channels.
        
    padding : P, optional, default: `v`
        The padding used for the convolutional layer. This attribute can be one of the following:
        
            * ``Ps``: The **SAME** padding mode.
            * ``Pv``: The **VALID** padding mode.
            * ``PXxY``: The value ``X`` is used for padding left and right and the value ``Y`` used for top and bottom.
            * ``PLxRxTxB``: The value ``L`` is used for left, ``R`` for right, ``T`` for top, and ``B`` for bottom.
        
    hasBias : B, optional, default: 1
        This attribute indicates whether a bias is used for this convolutional layer. If this is 1 (the default), a bias vector is used in this layer. Otherwise if ``B0`` is included in this *layerInfo*, it means the bias is not used.

    dilation : D, optional, default: 1
        The dilation for the convolutional layer. If the dilation is the same for both dimensions, only one integer value is enough to specify the dilation. Otherwise the dilation along the width and height are included and separated by 'x'. For example ``D2x4`` specifies a dilation of 2 along the width and 4 along height.

    rank : R, optional, default: 0
        This is used for Low-Rank models. Low-Rank decomposition is an algorithm used by Fireball to reduce the number of parameters of a model. If this layer is a low-rank decomposed layer, the rank attribute is a positive number specifying the rank of the weight tensor. Otherwise for regular models, this is set to 0 which is the default. In most cases this should be left unchanged when composing *layersInfo*. The method :py:meth:`~fireball.model.Model.createLrModel` can be used to reduce the number of parameters of the model. When this method is called, it automatically assigns a `rank` value for each decomposed layer.

Here are some examples::

    # Kernel size 3x3, stride 2 along width and 1 along height,
    # 128 output channels, "SAME" padding
        "CONV_K3_S2x1_O128_Ps"
        
    # Kernel size 5x3 or shape (3,5), stride 1, 128 output channels,
    # padding: Left: 2, right: 3, top: 1, bottom: 1
        "CONV_K5x3_O128_P2x3x1x1"
        
    # Kernel size 3x3, stride 1, dilation 6, 1024 output
    # channels, "SAME" padding, No biases
        "CONV_K3_D6_O1024_Ps_B0"
        
.. _DWCN:

DWCN: Depth-Wise Convolutional Layer
------------------------------------
This layer implements a depth-wise convolution operation on the input tensor.

Attributes
^^^^^^^^^^
    kernel : K
        The kernel size for this layer. For square kernels, only one integer value is enough to specify the kernel size. For example ``K3`` specifies a 3x3 kernel. For non-square kernels, the width and height of the kernel are included and separated by 'x'. For example ``K3x5`` specifies a 3x5 kernel. Please note that the actual *shape* of kernel is (5,3) in this case. (5 rows, 3 columns)
        
    stride : S, optional, default: 1
        The stride of convolution. If the stride is the same for both dimensions, only one integer value is enough to specify the stride. Otherwise the stride along the width and height are included and separated by 'x'. For example ``S2x1`` specifies a stride of 2 along the width and 1 along height.
        
    padding : P, optional, default: `v`
        The padding used for the convolutional layer. This attribute can be one of the following:
        
            * ``Ps``: The **SAME** padding mode.
            * ``Pv``: The **VALID** padding mode.
            * ``PXxY``: The value ``X`` is used for padding left and right and the value ``Y`` used for top and bottom.
            * ``PLxRxTxB``: The value ``L`` is used for left, ``R`` for right, ``T`` for top, and ``B`` for bottom.
        
    hasBias : B, optional, default: 1
        This attribute indicates whether a bias is used for this convolutional layer. If this is 1 (the default), a bias vector is used in this layer. Otherwise if ``B0`` is included in this *layerInfo*, it means the bias is not used.

    rank : R, optional, default: 0
        This is used for Low-Rank models. Low-Rank decomposition is an algorithm used by Fireball to reduce the number of parameters of a model. If this layer is a low-rank decomposed layer, the rank attribute is a positive number specifying the rank of the weight tensor. Otherwise for regular models, this is set to 0 which is the default. In most cases this should be left unchanged when composing *layersInfo*. The method :py:meth:`~fireball.model.Model.createLrModel` can be used to reduce the number of parameters of the model. When this method is called, it automatically assigns a `rank` value for each decomposed layer.

Here is an example::

    # Kernel size 3x3, stride 1, "SAME" padding, no biases
        "DWCN_K3_S1_Ps_B0"

.. _BN:

BN: Batch Normalization Layer
-----------------------------
This layer implements a batch normalization operation on the input tensor.

Attributes
^^^^^^^^^^
    epsilon : E, optional, default: 0.001
        The epsilon value used to prevent division by zero in calculations.
    
.. _LN:

LN: Layer Normalization Layer
-----------------------------
This layer implements a layer normalization operation on the input tensor.

Attributes
^^^^^^^^^^
    epsilon : E, optional, default: 1.0e-12
        The epsilon value used to prevent division by zero in calculations.

.. _AFM:

AFM: Aggregate Feature Maps Layer
---------------------------------
This layer is used in object detection models (such as SSD). It gathers the feature maps from outputs of different layers and uses internal convolutional layers to calculate the predicted classes and box adjustments for each anchor box.

The output of a layer is marked as a feature map using the :ref:`FM <FM>` post-activation.

Attributes
^^^^^^^^^^
    numClasses : C
        Number of classes for the object detection model including the background class

Here is an example of how this layer works with :ref:`FM <FM>` post-activations. This is the SSD-512 object detection model::

    layersInfo = 'IMG_S512_D3                                                      \
                  CONV_K3_O64_Ps:ReLu,CONV_K3_O64_Ps:ReLu:MP_K2_Ps                 \
                  CONV_K3_O128_Ps:ReLu,CONV_K3_O128_Ps:ReLu:MP_K2_Ps               \
                  2*CONV_K3_O256_Ps:ReLu,CONV_K3_O256_Ps:ReLu:MP_K2_Ps             \
                  2*CONV_K3_O512_Ps:ReLu,CONV_K3_O512_Ps:ReLu:FM_A4_N2:MP_K2_Ps    \
                  2*CONV_K3_O512_Ps:ReLu,CONV_K3_O512_Ps:ReLu:MP_K3_S1_Ps          \
                  CONV_K3_D6_O1024_Ps:ReLu,CONV_K1_O1024_Ps:ReLu:FM_A6             \
                  CONV_K1_O256_Ps:ReLu,CONV_K3_S2_O512_Ps:ReLu:FM_A6               \
                  CONV_K1_O128_Ps:ReLu,CONV_K3_S2_O256_Ps:ReLu:FM_A6               \
                  CONV_K1_O128_Ps:ReLu,CONV_K3_S2_O256_Ps:ReLu:FM_A6               \
                  CONV_K1_O128_Ps:ReLu,CONV_K3_S2_O256_Ps:ReLu:FM_A4               \
                  CONV_K1_O128_Ps:ReLu,CONV_K2_S2_O256_Ps:ReLu:FM_A4               \
                  AFM_C81                                                          \
                  OBJECT'

The model in the above example has 7 feature maps. The first feature map uses L2 normalization. The AFM layer near the end has 81 classes (80 plus one for background) and the :ref:`OBJECT<OBJECT>` output layer is used.

.. _BERT:

BERT: BERT Layer
----------------
This layer implements a Bidirectional Encoder Representations from Transformers layer. This implementation is based on `Google's original BERT model <https://github.com/google-research/bert>`_.

Attributes
^^^^^^^^^^
    outSize : O
        The output size of this layer. This is also known as the *hidden size* of a BERT layer.
        
    intermediateSize : I
        The intermediate size of BERT layer.

    numHeads : H, optional, default: 12
        The number of heads for the BERT layer. The default is 12.

    dropRate : R, optional, default: 0.1
        Internally a BERT layer uses some drop-out operations. This attribute gives the drop-rate for these drop-out operations. The default is 0.1.

    initSdt : S, optional, default: 0.02
        This is the Standard Deviation of the distribution used for random initialization of weight parameters in this layer.
        
    epsilon : E, optional, default: 1.0e-12
        The epsilon value used in the internal layer normalizations.
        
.. _CLASS:

CLASS: Classification Output Layer
----------------------------------
This output layer is used for classification models. It implements the computation of loss function for training and the predicted probabilities of classes for inference.

Attributes
^^^^^^^^^^
    numClasses : C
        The number of classes for the classification model.

.. note::

    Since this layer includes the softmax function (for multi-class classification) or sigmoid function (for binary classification), there is no need to add these activation functions to the last Fully Connected layer of the model (The one just before this output layer).

.. _REG:

REG: Regression Output Layer
----------------------------
This output layer is used for regression models. The output of the model can be a floating point scaler value or a tensor with floating values.

Attributes
^^^^^^^^^^
    shape : S, optional, default: 0
        The shape of output. The default is 0 which means a scaler output.

Here are some examples::

    # Scaler output
        "REG_S0"

    # The output is a vector of size 4
        "REG_S4
        
    # The output is an RGB image of size 32x32
        "REG_S32/32/3"

.. _OBJECT:

OBJECT: Object Detection Output Layer
-------------------------------------
This output layer is used for object detection models. It usually follows an :ref:`AFM <AFM>` layer.

This layer does not have any attributes.

.. _ANSWER:

ANSWER: Answer Output Layer
---------------------------
This output layer is used for question-answering models (such as the model for SQuAD). It outputs the start and end indexes of the predicted answer in a given context for a given question.

This layer does not have any attributes.

.. _RS:

RS: Reshape
-----------
This post-activation reshapes the output of current layer to the specified shape. The specified shape must be compatible with current shape of layer output.

Attributes
^^^^^^^^^^
    shape : S
        The shape of output. The values for different dimensions of the output tensor are separated by '/'.

Here is an example::

    # Reshape the output of the fully connected layer to matrixes of 4x64
        "FC_O256:ReLU:RS_S4/64"

.. _MP:

MP: Max Pooling
---------------
This post-activation implements the Max Pooling operation on the output of convolutional layers.

Attributes
^^^^^^^^^^
    kernel : K
        The kernel size for Max Pooling. For square kernels, only one integer value is enough to specify the kernel size. For example ``K3`` specifies a 3x3 kernel. For non-square kernels, the width and height of the kernel are included and separated by 'x'. For example ``K3x5`` specifies a 3x5 kernel. Please note that the actual *shape* of kernel is (5,3) in this case. (5 rows, 3 columns)
        
    stride : S, optional, default: Same as kernel
        The stride for the Max Pooling. If the stride is the same for both dimensions, only one integer value is enough to specify the stride. Otherwise the stride along the width and height are included and separated by 'x'. For example ``S2x1`` specifies a stride of 2 along the width and 1 along height. If stride is not specified for Max Pooling, the default behavior is to use the same value as kernel.

    padding : P, optional, default: `v`
        The padding used for the Max Pooling. This attribute can be one of the following:
        
            * ``Ps``: The **SAME** padding mode.
            * ``Pv``: The **VALID** padding mode.
            * ``PXxY``: The value ``X`` is used for padding left and right and the value ``Y`` used for top and bottom.
            * ``PLxRxTxB``: The value ``L`` is used for left, ``R`` for right, ``T`` for top, and ``B`` for bottom.

Here is an example::

    # Kernel size 3x3, stride 2 along width and 1 along height,
    # 128 output channels, "SAME" padding, ReLU activation, Max pooling
    # with kernel 2x2 and stride 2x2.
        "CONV_K3_S2x1_O128_Ps:ReLU:MP_K2"

.. _AP:

AP: Average Pooling
-------------------
This post-activation implements the Average Pooling operation on the output of convolutional layers.

Attributes
^^^^^^^^^^
    kernel : K
        The kernel size for Average Pooling. For square kernels, only one integer value is enough to specify the kernel size. For example ``K3`` specifies a 3x3 kernel. For non-square kernels, the width and height of the kernel are included and separated by 'x'. For example ``K3x5`` specifies a 3x5 kernel. Please note that the actual *shape* of kernel is (5,3) in this case. (5 rows, 3 columns)
        
    stride : S, optional, default: Same as kernel
        The stride for the Average Pooling. If the stride is the same for both dimensions, only one integer value is enough to specify the stride. Otherwise the stride along the width and height are included and separated by 'x'. For example ``S2x1`` specifies a stride of 2 along the width and 1 along height. If stride is not specified for Average Pooling, the default behavior is to use the same value as kernel.

    padding : P, optional, default: `v`
        The padding used for the Max Pooling. This attribute can be one of the following:
        
            * ``Ps``: The **SAME** padding mode.
            * ``Pv``: The **VALID** padding mode.
            * ``PXxY``: The value ``X`` is used for padding left and right and the value ``Y`` used for top and bottom.
            * ``PLxRxTxB``: The value ``L`` is used for left, ``R`` for right, ``T`` for top, and ``B`` for bottom.

Here is an example::

    # Kernel size 3x3, stride 2 along width and 1 along height,
    # 128 output channels, "SAME" padding, ReLU activation, Average pooling
    # with kernel 2x2 and stride 2x2.
        "CONV_K3_S2x1_O128_Ps:ReLU:AP_K2"

.. _TP:

TP: Transformer pooling
-----------------------
This post-activation is used in transformer models. It is usually used on the output of the last :ref:`BERT<BERT>` layer. It uses the first `n` vectors from the output sequence of the BERT layer.

Attributes
^^^^^^^^^^
    numVectors : N, optional, default: 1
        The number of vectors to include as the output of the BERT layer. The default is 1 which uses only the first vector in the sequence.

This is usually used to feed the fully connected layer that follows the last BERT layer for text classification applications.

.. _GAP:

GAP: Global Average Pooling
---------------------------
This post-activation implements the Global Average Pooling on the output of a convolutional layer.

This post-activation does not have any attributes.

.. _UP:

UP: Upsampling
--------------
This post-activation implements the Upsampling operation on the output of a convolutional layer.

Attributes
^^^^^^^^^^
    scale : S
        The scale for upsampling. If the scale is the same for both dimensions, only one integer value is enough to specify the scale. Otherwise the scale along the width and height are included and separated by 'x'. For example ``S2x4`` specifies a scale of 2 along the width and 4 along height.

.. _DO:

DO: Dropout
-----------
This post-activation implements the drop-out operation on the output of a fully connected or convolutional layer.

Attributes
^^^^^^^^^^
    dropRate : R, optional, default: global drop rate
        The rate or probability of drop out. Fireball allows the drop rate to be specified for each layer or globally for the whole model. The combination of the drop-rate values specified for each `DO` operation and the global drop-rate determines the drop-out behavior as follows:
        
            * If the drop-rate for the whole model is 1.0 (that is dropOutKeep=0, see Model's :py:meth:`~fireball.model.Model.__init__` method), then the dropout is globally disabled. The drop-rate values specified for DO post-activations are all ignored in this case.
            
            * If the drop-rate for the whole model is 0.0 (that is dropOutKeep=1 which is the default, see Model's :py:meth:`~fireball.model.Model.__init__` method), then drop out is disabled for all the DO post-activations without a specified drop-rate value. All other DO operations use their specified drop-rate values.
            
            * Otherwise, if the drop-rate for the whole model is a number between 0 and 1, this rate is used for all the DO post-activations without a specified drop-rate. All other DO operations use their specified drop-rate values.

.. _CLP:

CLP: Clip
---------
This post-activation clips the output of a layer to the specified min and max values.

Attributes
^^^^^^^^^^
    hiVal : H, optional, default: inf
        The maximum value to clip to.
        
    loVal : L, optional, default: -inf
        The minimum value to clip to.
        
At least one of hiVal or loVal must be specified.

Here is an example::

    # A Batch Normalization layer, ReLU activation, clipped
    # to a maximum of 6.0. Taken from MobileNetV2.
    # This is how a "ReLU6" can be implemented in Fireball
        "BN:ReLU:CLP_H6"

.. _L2R:

L2R: L2 Regularization
----------------------
The L2 Regularization post-activation doesn't actually change the output of a layer. It just marks the parameters of the layer to be included in the calculation of L2 regularization.

Attributes
^^^^^^^^^^
    factor : F, optional, default: 1.0
        The factor applied to the L2 norm of the parameters of this layer. Fireball allows the L2 Regularization factor to be specified for each layer or globally for the whole model. The combination of the factor specified for each `L2R` post-activation and the global regularization factor determines the regularization behavior as follows:
        
            * If the regularization factor for the whole model is 0.0 (that is regFactor=0 which is the default, see Model's :py:meth:`~fireball.model.Model.__init__` method), then L2 Regularization is globally disabled. The `factor` values specified for `L2R` post-activations are all ignored in this case.
            
            * Otherwise, if the regularization factor for the whole model is non-zero, this global value is used for all L2R post-activations without a factor specified. The L2R post-activations with a factor specified use their own factor.

The actual L2 regularization value for the whole model is the summation of L2 norms of all parameters of the layers with an L2R post-activation, weighted by the factor values as specified above.

.. _FM:

FM: Feature Map
---------------
The Feature Map post-activation is used with object detection models to specify the output of a layer as a Feature Map. An :ref:`AFM<AFM>` layer near the end of the *layersInfo* is then used to "aggregate" these feature maps and create the class and box predictions for the objects detected in the image.

Attributes
^^^^^^^^^^
    anchors : A
        The number of "Anchor boxes" for the feature map.
        
    norm : N, optional, default: 0
        This attribute specifies the type of normalization applied to the feature maps. Currently only 0 and 2 are the supported values. A value of 0 (the default) means there is no normalization for this feature map. A value of 2 means L2 normalization should be applied to this feature map. The :ref:`AFM<AFM>` layer uses this information when combining all feature maps.
        
Please refer to the documentation of :ref:`AFM<AFM>` layer for more information and an example of FM post-activations used in an object detection model.

.. _ADD:

ADD: Add netmarks
-----------------
This post-activations adds the output this layer with the specified :ref:`netmarks<NETMARK>`.

Attributes
^^^^^^^^^^
    netmarks : N
        The netmark IDs that are added to the output of current layer. At least one netmark must be specified. Multiple netmark IDs are separated by '/'.
        

Here is an example::

    # Add the output of this convolutional layer with
    # netmarks 2 and 3 (which should be defined somewhere in
    # previous layers).
    "CONV_K3_O128:ReLU:ADD_N2/3"

.. _SEL:

SEL: Select netmark
-------------------
This post-activations selects and outputs one of the specified :ref:`netmarks<NETMARK>` based on the output of this layer. This post-activation can only be used with :ref:`FC<FC>` layers.

Attributes
^^^^^^^^^^
    netmarks : N
        The netmark IDs to choose from. This layer's output determines which one of the specified netmarks is used as the output of this layer. Multiple netmark IDs are separated by '/'. At least 2 netmark ID are required for this post-activation to work.
        
If only 2 netmark are specified (binary selection), this layer must be a fully connected layer with output size 1 and Sigmoid activation function. If sigmoid's output value is less than 0.5 the first netmark value is used as output, otherwise the second one is used.

Otherwise, if there are `n` netmarks (`n`>2), this layer must be a fully connected layer with output size of `n`. In this case the i\ :sup:`th` netmark is used for the output of this layer where `i = argmax(FC layer output)`.

.. note::

   Since this operation is not differentiable, this post-activation can only be used for inference. A common use case is to train different sub-models separately and then "merge" them together using this post-activation to make a larger model for inference.

Here are a couple of examples::

    # The output of this layer is netmark ID 2 if the
    # output of layer (the output of sigmoid function) is
    # less than 0.5, and netmark ID 3 otherwise.
    # Netmarks 2 and 3 must have been define somewhere in
    # previous layers.
    "FC_O1:Sig:SEL_N2/3"

    # Assuming the output of fully connected layer is "fcOut",
    # output of this layer is:
    #   netmark 3, if fcOut[0] is the largest value in fcOut
    #   netmark 4, if fcOut[1] is the largest value in fcOut
    #   netmark 7, if fcOut[2] is the largest value in fcOut
    #   netmark 9, if fcOut[3] is the largest value in fcOut
    # Netmarks 3,4,7, and 9 must have been define somewhere in
    # previous layers.
    "FC_O4::SEL_N3/4/7/9"
    
.. _WSUM:

WSUM: Weighted Sum Netmarks
---------------------------
This post-activations outputs a weighted summation of the specified :ref:`netmarks<NETMARK>`. The summation is weighted by the output of this layer. This post-activation can only be used with :ref:`FC<FC>` layers and sigmoid or softmax activations.

Attributes
^^^^^^^^^^
    netmarks : N
        The netmark IDs that are included in the weighted sum. The output of activation function in this layer determines the weights for each one of the specified netmarks. Multiple netmark IDs are separated by '/'. At least 2 netmark ID are required for this post-activation to work.
        
If only 2 netmarks are specified, this layer must be a fully connected layer with output size 1 and Sigmoid activation function. Assuming sigmoid's output value is `sigOut` the output of this layer is:

.. math:: output = sigOut.netmark_1 + (1-sigOut).netmark_0
    
where netmark\ :sub:`0` and netmark\ :sub:`1` are the values of the first and second netmark specified in this post-activation.

Otherwise, if there are `n` netmarks (`n`>2), this layer must be a fully connected layer with output size of `n` and Softmax activation function. Assuming softmax's output value is `weights` the output of this layer is:

.. math:: output = \sum_{i=0}^{n-1} weights[i].netmark_i

where netmark\ :sub:`0` to netmark\ :sub:`n-1` are the values of the `n` netmarks specified in this post-activation.

Here are a couple of examples::

    # For 2 netmarks, output size is 1 and Sigmoid is used
    "FC_O1:Sig:WSUM_N2/3"

    # For 3 netmarks, output size is 3 and Softmax is used
    "FC_O3:Soft:WSUM_N3/4/5"

.. _TUP:

TUP: Tuple netmarks
-------------------
This post-activations bundles the output of this layer with the specified :ref:`netmarks<NETMARK>` in a tuple and returns the whole tuple as the output of this layer.

Attributes
^^^^^^^^^^
    netmarks : N
        The netmark IDs that are included in tuple. At least one netmark is required for this post-activation to work.


.. _RND:

RND: Round
----------
This post-activations rounds the output of this layer to the nearest integer value.

This post-activation does not have any attributes.

.. _NETMARK:

Netmarks
========
Most neural networks are made up of a sequence of layers where the output of each layer is fed to the input of following layer. However there are some cases where the network is a directed graph of layers. In other words there can be bypass paths (such as the ones in Residual Networks).

Fireball uses the concept of Netmarks to allow implementation of multiple paths. A netmark is a location in the network structure (such as output of a layer) that is remembered (like a bookmark in a book). Each netmark is specified by a unique integer number. To add the output of a layer to the model's list of netmarks, you can use the ">X" notation where 'X' is the unique identifier of the netmark. In the following example the output of a convolutional layer is added to the netmarks with a netmark ID of 1::

        # A Convolutional layer, 3x3 kernel, stride 2, ReLU activation
        # function, Max Pooling with Kernel and stride 2, the output
        # is added to netmarks with a netmark ID 1.
            "CONV_K3_S2:ReLU:MP_K2_S2>1"

Once a netmark is added to a model, it can be used in many different ways giving Fireball one of its powerful features. The first use case is when a netmark is specified as input to layer in the model.

In normal cases, the input to a layer is the output of the previous layer. But using the nemark input feature, you can specify one of the existing netmarks as the input to the layer. The "X>" notation before the layer text means the layer gets its input from the netmark specified by the netmark ID 'X'. In the following example the :ref:`Fully Connected layer<FC>` gets its input from netmark with ID 3::

        # A Fully Connected layer with 128 output channels with
        # Tangent Hyperbolic activation function, and drop-out with drop-rate 0.3
        # The input to this layer comes from the netmark with ID 3
            "3>FC_O128:Tanh:DO_R0.3"

Fireball also allows you to merge different paths using some special types of post-activations. For example, the following shows how to use the :ref:`ADD<ADD>` post-activation to add 2 different netmarks to the output of current layer before outputting it to the next layer::

        # A Convolutional layer, 3x3 kernel, stride 2, ReLU activation
        # function, Max Pooling with Kernel and stride 2, the output
        # is then added to the netmarks 3 and 4 and the results of the
        # addition is outputted to next layer.
            "CONV_K3_S2:ReLU:MP_K2_S2:ADD_N3/4"


.. _BLOCKS:

Blocks
======
Fireball blocks are like macros. You can define a combination of layers and connect them together as a block. The defined blocks can then be reused in the *layersInfo* just like any other fireball layer. A block is defined in a single text string called *blockInfo*. When creating a model, create a list of *blockInfo* strings and pass it to the :py:class:`fireball.model.Model` class.

A block can have up to 10 different paths that are merged together and used as the output of the block.

*blockInfo* Syntax:
--------------------
A *blockInfo* string has 4 main parts which are separated by  '|'::

    blockInfo = "<name>|<attributes>|<mergeFunc>|<pathLayersInfo1>;<pathLayersInfo2>;..."

Let's look at the details of each part:

    * name: This is the name of the block. This name is used when this block is "called" inside *layersInfo* just like the names of Fireball's native layers.
    * attributes: This part defines the attributes of the block. It is made up of zero or more attribute definitions separated by comma:
    
        .. code-block:: python
    
            attributes = attDef0,attDef1,...
        
        attribute definitions contain attribute letter, name, type, and default separated by underscore characters:
    
        * Attribute Letter: This is the letter used to specify the value of this attribute when this block is used in the *layersInfo*.
        * Attribute Name: The name of attribute.
        * Attribute Type: Currently the following type specifiers are supported:
            
            * ``i``: signed integer
            * ``u``: unsigned integer
            * ``ixi``: A pair of signed integers. Usually for 2D attributes. If one value is specified, it is assigned to both.
            * ``uxu``: A pair of unsigned integers. Usually for 2D attributes. If one value is specified, it is assigned to both.
            * ``f``: A floating point value.
            * ``p``: A padding type value. Similar to the `padding` attribute of :ref:`Convolutional<CONV>` layers.
            * ``b``: A boolean value.

        .. note::

            The attribute types ``i``, ``u``, ``f``, and ``b`` can be used to define *list attributes*. For example ``i*3`` means the attribute takes 3 integer values separated by "/" (See the ResNet50 example below). If the number of items in the list is not known, you can use ``i*?``.
            
        * Attribute Default: If a default is specified for an attribute, it means this is an optional attribute. When this block is used in the *layersInfo*, if a value is not specified for this attribute, the specified default value will be used. If a default value is not specified, it means this attribute is required and must be specified when it is used in the model's *layersInfo*.

    * mergeFunc: This is the function used to merge the outputs of different paths. Currently only "add" is supported which adds the outputs of all paths.
    * pathLayersInfo1;...: A block can have up to 10 paths. Each path is comma delimited list of Fireball native layers just like they are used in *layersInfo*. The only difference is that we can use the block attributes as place holders for the attributes of the layers in the path.
    
        .. note::
    
            A *pathLayersInfo* can be just a bypass route from the input to the output of block. In this case the pseudo layer **ID** (short for Identity) can be used to specify a bypass path (See the second example below)


Block Examples
--------------
Here are some examples (From MobileNetV2 and ResNet50) of how to define and use
Blocks::

    # Example 1:
    # This is one of the blocks used by MobileNetV2.
    # Block name is "MN1"
    # It has 2 attributes:
    #    extension: letter "X" is used to specify this attribute.
    #    outDept: letter "O" is used to specify this attribute.
    # Both attributes are integer (letter "i" is used) and required
    # because a default value is not specified.
    # This block has only one path (No semicolons used in the "pathLayersInfo")
    # Note how block attribute "%x" and "%o" are used in the "pathLayersInfo"
    # as place holders for the output size of the first and last convolutional
    # layers.
    # For example "CONV_K1_O%x_Ps_B0" means use whatever value passed
    # to the block as X (expansion) for the outDept (O) of this convolutional
    # layer.
    
    blockInfo = 'MN1|x_expansion_i,o_outDept_i|                     \
                 add|                                               \
                 CONV_K1_O%x_Ps_B0,BN:ReLU:CLP_H6,DWCN_K3_S1_Ps_B0, \
                    BN:ReLU:CLP_H6,CONV_K1_O%o_Ps_B0,BN'

    # This is an example of how this block is used in the "layersInfo" (from
    # MobileNetV2) with expansion=384 and outDept=96:
    
        layersInfo = "...,MN1_X384_O96,..."

    # Example 2:
    # Here is another similar MobileNetV2 block with a shortcut path. Note
    # the usage of "ID" for the second path:
    
    blockInfo = 'MN1S|x_expansion_i,o_outDept_i|                    \
                 add|                                               \
                 CONV_K1_O%x_Ps_B0,BN:ReLU:CLP_H6,DWCN_K3_Ps_B0,    \
                    BN:ReLU:CLP_H6,CONV_K1_O%o_Ps_B0,BN;ID'

    # Example 3:
    # Here is another example from ResNet50 with 2 paths and 3 attributes.
    # Note the usage of list of integers for the second attribute and
    # the default value for the third attribute. Also note how the items
    # in the list are used as place holders in "pathLayersInfo".
    # For example "CONV_K%k_S1_O%o1_Ps" means use the value of block attribute
    # "kernel" for the kernel (K%k) and the second item in the list of the block
    # attribute "outSizes" for the outDept (O%o1) of this convolutional layer.
    
    blockInfo = 'RES2|k_kernel_ixi,o_outSizes_i*3,s_stride_ixi_1    \
                 add|                                               \
                 CONV_K1_S%s_O%o0_Pv,BN:ReLU,CONV_K%k_S1_O%o1_Ps,   \
                    BN:ReLU,CONV_K1_S1_O%o2,BN;                     \
                 CONV_K1_S%s_O%o2_Pv,BN'

    # Here is an example how this block is used in the "layersInfo" of
    # ResNet50 with kernel=3, outSizes=[64,64,256], and stride=1
    
        layersInfo = "...,RES2_K3_O64/64/256_S1:ReLU,..."
