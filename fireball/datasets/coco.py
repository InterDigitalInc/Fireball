# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of `COCO <https://cocodataset.org>`_ dataset class for object detection. Use the ``CocoDSetUnitTest.py`` file in the ``UnitTest/Datasets`` folder to run the Unit Test of this implementation.

A sample when it is provided in `getBatch` is tuple of the form: (image, classes, boxes). Internally it is kept as a tuple of the form (imageId, classes, boxes, areas, crowdFlags).

    * ``imageId``: Unique id of the Image. (int32)
    * ``image``: A numpy float32 tensor of shape (h, w, 3)
    * ``classes``: A numpy int32 tensor of shape (n,) where n is the number of objects in the image. Each number is the class index of the object. Class indexes start at '1' and end with '80'. Class '0' is reserved for background.
    * ``boxes``: A numpy float32 tensor of shape (n,4) where n is the number of objects in the image. The 4 numbers for each bbox are x1, y1, w, and h. (``P1Size`` format)
    * ``areas``: A numpy float32 tensor of shape (n,) where n is the number of objects in the image. Each number give the area of the object (IMPORTANT NOTE: This is different and usually smaller than the area of the box)
    * ``crowdFlags``: A numpy boolean tensor of shape (n,). Each element indicates whether the object is in a crowd of other overlapping objects.

A batch of samples is a list of samples as defined above.

This implementation assumes the following files/folders exist in the `dataPath` directory:

    * ``annotations``: The json files containing annotations (information about images, and objects inside each image)
    * ``train2014``: Training Images (2014)
    * ``val2014``: Validation Images (2014)
    * ``val2017``: Validation Images (2017)

**Dataset Stats**
    +-----------+----------+-----------+---------------------+--------------+--------------+
    |           |  Total   |  Total    |        Crowd        | Images with  | Max Objects  |
    | Dataset   |          |           +-----------+---------+              |              |
    |           |  Images  |  Objects  |  Objects  |  Images | no Objects   | Per Image    |
    +===========+==========+===========+===========+=========+==============+==============+
    | train2014 |  82,783  |  604,906  |  7,038    |  6,395  |  702         |  93          |
    +-----------+----------+-----------+-----------+---------+--------------+--------------+
    | val2014   |  40,504  |  291,874  |  3,460    |  3131   |  367         |  70          |
    +-----------+----------+-----------+-----------+---------+--------------+--------------+
    | val2017   |  5,000   |  36,781   |  446      |  411    |  48          |  63          |
    +-----------+----------+-----------+-----------+---------+--------------+--------------+
"""

# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 03/02/2020              Shahab Hamidi-Rad       Created the file.
# 04/17/2020              Shahab Hamidi-Rad       Changed the constructor signature to match the other
#                                                 datasets.
# 04/21/2020              Shahab Hamidi-Rad       Completed the documentation.
# 04/23/2020              Shahab Hamidi-Rad       Now using daemonic threads.
# 10/11/2021              Shahab Hamidi-Rad       Added support for downloading datasets.
# **********************************************************************************************************************
import numpy as np
import os
from threading import Thread
import traceback
import time
import cv2
import json
from .base import BaseDSet
from ..printutils import myPrint

plt = None      # matplotlib.pyplot is imported later only if it is needed.

# **********************************************************************************************************************
class CocoDSet(BaseDSet):
    r"""
    This class implements the Coco Dataset dataset.
    """
    numClasses = None
    classId2Idx = None
    
    # For this dataset we only use cafe style preprocessing:
    #   * Images are in BGR format as they are fed to the network
    #   * They are normalized using the mean values: 103.939, 116.779, 123.68 for blue, green, and red.
    #   * The image tensor is then returned as float32 numbers.
    bgrMeans = np.float32([103.939, 116.779, 123.68])
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, batchSize=64, resolution=512, keepAr=True, numWorkers=4):
        r"""
        Constructs an CocoDSet instance. This can be called directly or via `makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", "Valid".

        dataPath : str
            The path to the directory where the dataset files are located.

        batchSize : int
            The default batch size used in the "batches" method.

        resolution : int
            The resolution of the images. Default is 512 for 512x512 images

        keepAr : Boolean
            This specifies whether the aspect ratio of the image should be kept when it is resized.

        numWorkers : int
            The number of worker threads used to load the images.
        """

        if dataPath is None:
            dataPath = '/data/mscoco/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        if 'train' in dsName.lower():   self.cocoDs = 'train2014'
        elif 'test' in dsName.lower():  self.cocoDs = 'val2017'
        elif 'valid' in dsName.lower(): self.cocoDs = 'val2014'
        else:                           raise ValueError("Unknown dataset name \"%s\"!"%(dsName))

        self.dsName = dsName
        self.dataPath = dataPath
        self.labels = None
        self.batchSize = batchSize
        self.resolution=resolution
        self.keepAr=keepAr
        self.numWorkers=numWorkers
        self.augment=False
        
        dataDic = json.load( open( dataPath + 'annotations/instances_' + self.cocoDs + '.json' ) )
        
        imageIdToInfo = {}
        for imageInfo in dataDic['images']:
            imageIdToInfo[ imageInfo['id'] ] = ( imageInfo['id'], [], [], [], [] )  # imgId, classes, box, area, isCrowd
        
        categoryIds = set()
        for annotation in dataDic['annotations']:
            if annotation.get("ignore", 0) == 1:    continue
            categoryIds.add(annotation['category_id'])
            imgId = annotation['image_id']
            if imgId not in imageIdToInfo:          continue
            i, c, b, a, d = imageIdToInfo[ imgId ]
            if annotation['bbox'][2]==0.0 or annotation['bbox'][3]==0.0:    continue    # Ignore boxes with zero size
            imageIdToInfo[ imgId ] = (i, c+[annotation['category_id']], b+[annotation['bbox']],
                                         a+[annotation['area']], d+[annotation['iscrowd']])

        categoryIds = sorted(list(categoryIds))
        labels = ["background"] + ["" for _ in range(len(categoryIds))]
        for category in dataDic['categories']:
            if category['id'] not in categoryIds: continue
            labels[ categoryIds.index( category['id'] )+1 ] = category['name']

        for imgId, imgInfo in imageIdToInfo.items():
            i, c, b, a, d = imgInfo
            imageIdToInfo[ imgId ] = ( i, np.int32([categoryIds.index(cc)+1 for cc in c]), np.float32(b),
                                          np.float32(a), np.int32(d) )

        self.samples = list(imageIdToInfo.values())
        self.numSamples = len(self.samples)
        self.sampleShape = (resolution, resolution, 3)
        self.labelShape = None
        self.sampleIndexes = np.arange(self.numSamples)
        self.workToDo = None
        self.workDone = None

        self.imgIdToIndex = {}
        for s,sample in enumerate(self.samples):  self.imgIdToIndex[sample[0]] = s
        
        if CocoDSet.classNames is None:
            np.random.seed(1234)    # Do this once!
            CocoDSet.classNames = labels
            CocoDSet.classId2Idx = [-1]*(categoryIds[-1]+1)
            CocoDSet.numClasses = len(CocoDSet.classNames)
            for catIdx,catId in enumerate(categoryIds): CocoDSet.classId2Idx[ catId ] = catIdx+1
        else:
            assert CocoDSet.classNames == labels, "Class Names do not match with existing names!"

        if self.cocoDs == 'train2014':  fileNameTemplate = dataPath + 'train2014/COCO_train2014_%012d.jpg'
        elif self.cocoDs == 'val2014':  fileNameTemplate = dataPath + 'val2014/COCO_val2014_%012d.jpg'
        elif self.cocoDs == 'val2017':  fileNameTemplate = dataPath + 'val2017/%012d.jpg'
        self.idToFileName = lambda id: fileNameTemplate%id

        self.anchorBoxes = None
        self.anchorsBoxesP1P2 = None
        self.__class__.evalMetricName = 'mAP'

    # ******************************************************************************************************************
    @classmethod
    def download(cls, dataFolder=None):
        r"""
        This class method can be called to download the COCO dataset files.
        
        Parameters
        ----------
        dataFolder: str
            The folder where dataset files are saved. If this is not provided, then
            a folder named "data" is created in the home directory of the current user and the
            dataset folders and files are created there. In other words, the default data folder
            is ``~/data``
        """
        files = ['annotations.zip',
                 'http://images.cocodataset.org/zips/train2014.zip',
                 'http://images.cocodataset.org/zips/val2014.zip',
                 'http://images.cocodataset.org/zips/val2017.zip']
        BaseDSet.download("mscoco", files, dataFolder)

    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=64, dataPath=None, resolution=512, keepAr=True, numWorkers=4):
        r"""
        This class method creates several datasets as specified by `dsNames` parameter in one-shot.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
            
            * **"Train"**: Create the training dataset. Training dataset uses the images in the "train2014" folder.
            * **"Test"**:  Create the test dataset. Test dataset uses the images in the "val2017" folder.
            * **"Valid"**: Create the validation dataset. Validation dataset uses the images in the "val2014" folder.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        resolution : int
            The resolution of the images. Default is 512 for 512x512 images

        keepAr : Boolean
            This specifies whether the aspect ratio of the image should be kept when it is resized.

        numWorkers : int
            The number of worker threads used to load the images.
            
        Returns
        -------
        Up to 3 CocoDSet objects
            Depending on the number of items specified in the `dsNames`, it returns between one and three CocoDSet objects. The returned values have the same order as they appear in the `dsNames` parameter.

        Note
        ----
        * To specify the training dataset, any string containing the word "train" (case insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        * To specify the test dataset, any string containing the word "test" (case insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        * To specify the validation dataset, any string containing the word "valid" (case insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.

        Examples
        --------
        * ``dsNames="Train,Test,Valid"``: 3 `ImageNetDSet` objects are returned for training, test, and validation in the same order.

        * ``dsNames="TRAINING,TEST"``: 2 `ImageNetDSet` objects are returned for training and test.
        """
        
        if dataPath is None:
            dataPath = '/data/mscoco/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            retVals += [ cls(dsStr, dataPath, batchSize, resolution, keepAr, numWorkers) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals

    # ******************************************************************************************************************
    def __repr__(self):
        r"""
        Provides a text string containing the information about this instance of CocoDSet. Calls the base class implementation first and then adds the information specific to this class.
        
        Returns
        -------
        str
            The text string containing the information about instance of CocoDSet.
        """
        repStr = super().__repr__()
        repStr += '    Resolution ..................................... %dx%d\n'%(self.resolution,self.resolution)
        repStr += '    Keep Aspect Ratio .............................. %s\n'%('Yes' if self.keepAr else "No")
        repStr += '    Number of Workers .............................. %d\n'%(self.numWorkers)
        return repStr
       
    # ******************************************************************************************************************
    def getStats(self):
        r"""
        Returns some statistics about this instance of CocoDSet.
        
        Returns
        -------
        sampleCounts : numpy array
            This is 2d numpy array. ``sampleCounts[c][1]`` is the number of images in the whole dataset that contains an instance of class 'c'. ``sampleCounts[c][0]`` is the total number of times instances of class 'c' appear in all images in the dataset.
        
        numCrowd : list
            This is a list containing 2 integer numbers. ``numCrowd[0]`` is the total number of times a "Crowd" object appears in the whole dataset. ``numCrowd[1]`` is the number of images in the dataset that contain at least one "Crowd" object.
            
        numEmptyImages : int
            This is the number of images in the dataset that don't contain any objects in it.
            
        maxObjectsPerImage : int
            This is the maximum number of objects that appeared in an image in the whole dataset.
        """
        sampleCounts = np.zeros((CocoDSet.numClasses,2))
        maxObjectsPerImage = 0
        numEmptyImages = 0
        numCrowd = [0, 0]
        for sample in self.samples:
            classesInSample = set(sample[1])
            for c in classesInSample:   sampleCounts[c, 1] += 1
            for c in sample[1]:         sampleCounts[c, 0] += 1
            if len(sample[1]) == 0:     numEmptyImages += 1
            if len(sample[1])>maxObjectsPerImage:    maxObjectsPerImage = len(sample[1])
            nc = sample[4].sum()
            numCrowd[0] += nc
            numCrowd[1] += 1 if nc>0 else 0
            
        return sampleCounts, numCrowd, numEmptyImages, maxObjectsPerImage
    
    # ******************************************************************************************************************
    @classmethod
    def printStats(cls, trainDs=None, testDs=None, validDs=None):
        r"""
        This class method prints statistics of classes for the given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : CocoDSet, optional
            The training dataset.
        testDs : CocoDSet, optional
            The test dataset.
        validDs : CocoDSet, optional
            The validation dataset.
        """
        
        print('\n    Dataset Statistics:')
        
        maxClassWidth = 0
        for className in cls.classNames:
            if len(className)>maxClassWidth:
                maxClassWidth = len(className)

        #              | 123 1234567890123 |
        sep =     '    +-----' + '-'*maxClassWidth + '-+'
        row1Str = '    | Class' + ' '*maxClassWidth + '|'
        row2Str = '    |      ' + ' '*maxClassWidth + '|'

        if trainDs is not None:
            sep +=       '--------------------+'
            row1Str +=   '       Train        |'
            row2Str +=   ' Objects    Images  |'
        #                | 123456789  1234567 |
            trainSampleCounts, trainCrowds, trainEmptyImages, trainMaxObjectsPerImage = trainDs.getStats()

        if validDs is not None:
            sep +=       '--------------------+'
            row1Str +=   '     Validation     |'
            row2Str +=   ' Objects    Images  |'
            validSampleCounts, validCrowds, validEmptyImages, validMaxObjectsPerImage = validDs.getStats()

        if testDs is not None:
            sep +=       '--------------------+'
            row1Str +=   '        Test        |'
            row2Str +=   ' Objects    Images  |'
            testSampleCounts, testCrowds, testEmptyImages, testMaxObjectsPerImage = testDs.getStats()

        print(sep)
        print(row1Str)
        print(row2Str)
        print(sep)

        for c in range(cls.numClasses):
            rowStr = ('    | %%3d %%-%ds |'%(maxClassWidth))%(c, cls.classNames[c])
            if trainDs is not None: rowStr += ' %-9d  %-7d |'%(trainSampleCounts[c,0], trainSampleCounts[c,1])
            if validDs is not None: rowStr += ' %-9d  %-7d |'%(validSampleCounts[c,0], validSampleCounts[c,1])
            if testDs is not None:  rowStr += ' %-9d  %-7d |'%(testSampleCounts[c,0], testSampleCounts[c,1])
            print( rowStr )

        print(sep)
        rowStr = '    | Total              |'
        if trainDs is not None:     rowStr += ' %-9d  %-7d |'%(sum(trainSampleCounts[:,0]), trainDs.numSamples)
        if validDs is not None:     rowStr += ' %-9d  %-7d |'%(sum(validSampleCounts[:,0]), validDs.numSamples)
        if testDs is not None:      rowStr += ' %-9d  %-7d |'%(sum(testSampleCounts[:,0]), testDs.numSamples)
        print( rowStr )

        rowStr = '    | Crowded            |'
        if trainDs is not None:     rowStr += ' %-9d  %-7d |'%(trainCrowds[0], trainCrowds[1])
        if validDs is not None:     rowStr += ' %-9d  %-7d |'%(validCrowds[0], validCrowds[1])
        if testDs is not None:      rowStr += ' %-9d  %-7d |'%(testCrowds[0], testCrowds[1])
        print( rowStr )
        
        print(sep)
        rowStr = '    | Empty Images       |'
        if trainDs is not None:     rowStr += ' %-18d |'%(trainEmptyImages)
        if validDs is not None:     rowStr += ' %-18d |'%(validEmptyImages)
        if testDs is not None:      rowStr += ' %-18d |'%(testEmptyImages)
        print( rowStr )

        rowStr = '    | Max obj. Per Image |'
        if trainDs is not None:     rowStr += ' %-18d |'%(trainMaxObjectsPerImage)
        if validDs is not None:     rowStr += ' %-18d |'%(validMaxObjectsPerImage)
        if testDs is not None:      rowStr += ' %-18d |'%(testMaxObjectsPerImage)
        print( rowStr )
        print(sep)

    # MARK: -------- Image and Bounding Box Manipulations --------
    # ******************************************************************************************************************
    def getImage(self, img):
        r"""
        This returns an image in BGR format as a numpy array of type float32 and shape (h,w,3).
        
        Parameters
        ----------
        img : numpy array, int/np.int32, or str
            
            * If this is a numpy array, it is assumed that the image has already been loaded and it is just returned without any modifications.
            * If this is an int/np.int32, it is assumed to be the id of the image and it is used to get the image file name and then load the image from the file.
            * If this is a str, then it is assumed to be the name of the file and it is used to load the image.
            
        Returns
        -------
        numpy array
            The loaded image is returned in BGR format as a numpy array of type float32 and shape (h,w,3). Where 'w' and 'h' are equal to the 'resolution' argument in the '__init__' or 'makeDatasets' functions.
        """
        
        if type(img) == np.ndarray:         return img      # Image already loaded
        if type(img) in [int, np.int32]:    img = self.idToFileName(img)
        assert type(img) == str, "img cannot be a '%s'!"%(type(img))
        
        # Reads image in BGR order
        imgFileName = img
        if os.path.exists(imgFileName):
            img = cv2.imread(imgFileName)
            assert img is not None, "Could not load image! (%s)"%(imgFileName)
        elif os.path.exists(self.dataPath + imgFileName):
            img = cv2.imread(self.dataPath + imgFileName)
            assert img is not None, "Could not load image! (%s)"%(self.dataPath + imgFileName)
        else:
            assert False, "Could not load image! (%s)"%(imgFileName)
        return np.float32(img)

   # *******************************************************************************************************************
    @classmethod
    def p1P2ToP1Size(cls, boxes):
        r""" Convert from [x1, y1, x2, y2] to [x1, y1, w, h]
        
        This class method changes all the boxes in the "boxes" array from "P1P2" format to "P1Size" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "P1P2" format. In "P1P2" format each box is represented with array of 4 numbers [x1, y1, x2, y2] where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners of the box correspondingly.
    
        Returns
        -------
        same shape and type of the input
            The box(s) in the "P1Size" format. In "P1Size" format each box is represented with array of 4 numbers [x1, y1, w, h] where (x1,y1) and (w,h) represent the top-left corner of the box and size of the box correspondingly.
        """
        
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([boxes[0], boxes[1], boxes[2]-boxes[0], boxes[3]-boxes[1]])
        
        p1Size = np.float32(boxes.copy())
        p1Size[:,2:] -= p1Size[:,:2]
        return p1Size

    # ******************************************************************************************************************
    @classmethod
    def p1SizeToP1P2(cls, boxes):
        r""" Convert from [x1, y1, w, h] to [x1, y1, x2,y 2]
        
        This class method changes all the boxes in the "boxes" array from "P1Size" format to "P1P2" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "P1Size" format. In "P1Size" format each box is represented with array of 4 numbers [x1, y1, w, h] where (x1,y1) and (w,h) represent the top-left corner of the box and size of the box correspondingly.
    
        Returns
        -------
        same shape and type of the input
            The box(s) in the "P1P2" format. In "P1P2" format each box is represented with array of 4 numbers [x1, y1, x2, y2] where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners of the box correspondingly.
        """
        
        if len(boxes) == 0: return boxes
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([boxes[0], boxes[1], boxes[2]+boxes[0], boxes[3]+boxes[1]])
        
        p1P2 = boxes.copy()
        p1P2[:,2:] += p1P2[:,:2]
        return p1P2

    # ******************************************************************************************************************
    @classmethod
    def p1P2ToCenterSize(cls, boxes):
        r""" Convert from [x1, y1, x2, y2] to [cx, cy, w, h]
        
        This class method changes all the boxes in the "boxes" array from "P1P2" format to "CenterSize" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "P1P2" format. In "P1P2" format each box is represented with array of 4 numbers [x1, y1, x2, y2] where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners of the box correspondingly.

        Returns
        -------
        same shape and type of the input
            The box(s) in the "CenterSize" format. In "CenterSize" format each box is represented with array of 4 numbers [cx, cy, w, h] where (cx,cy) and (w,h) represent the center point and size of the box correspondingly.
        """
        
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([(boxes[0]+boxes[2])/2.0, (boxes[1]+boxes[3])/2.0, boxes[2]-boxes[0], boxes[3]-boxes[1]])
        
        centerSize = boxes.copy()
        center = (boxes[:,2:] + boxes[:,:2])/2.0
        centerSize[:,2:] -= centerSize[:,:2]
        centerSize[:,:2] = center
        return centerSize

    # ******************************************************************************************************************
    @classmethod
    def centerSizeToP1P2(cls, boxes):
        r""" Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        
        This class method changes all the boxes in the "boxes" array from "CenterSize" format to "P1P2" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "CenterSize" format. In "CenterSize" format each box is represented with array of 4 numbers [cx, cy, w, h] where (cx,cy) and (w,h) represent the center point and size of the box correspondingly.

        Returns
        -------
        same shape and type of the input
            The box(s) in the "P1P2" format. In "P1P2" format each box is represented with array of 4 numbers [x1, y1, x2, y2] where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners of the box correspondingly.
        """
        
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([boxes[0]-boxes[2]/2.0, boxes[1]-boxes[3]/2.0, boxes[0]+boxes[2]/2.0, boxes[1]+boxes[3]/2.0])

        p1P2 = boxes.copy()
        p1P2[:,:2] -= p1P2[:,2:]/2.0
        p1P2[:,2:] += p1P2[:,:2]
        return p1P2

    # ******************************************************************************************************************
    @classmethod
    def p1SizeToCenterSize(cls, boxes):
        r""" Convert from [x1, y1, w, h] to [cx, cy, w, h]
        
        This class method changes all the boxes in the "boxes" array from "P1Size" format to "CenterSize" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "P1Size" format. In "P1Size" format each box is represented with array of 4 numbers [x1, y1, w, h] where (x1,y1) and (w,h) represent the top-left corner and size of the box correspondingly.

        Returns
        -------
        same shape and type of the input
            The box(s) in the "CenterSize" format. In "CenterSize" format each box is represented with array of 4 numbers [cx, cy, w, h] where (cx,cy) and (w,h) represent the center point and size of the box correspondingly.
        """
        
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([boxes[0]+boxes[2]/2.0, boxes[1]+boxes[3]/2.0, boxes[2], boxes[3]])

        centerSize = boxes.copy()
        centerSize[:,:2] += boxes[:,2:]/2
        return centerSize

    # ******************************************************************************************************************
    @classmethod
    def centerSizeToP1Size(cls, boxes):
        r""" Convert from [cx, cy, w, h] to [x1, y1, w, h]
        
        This class method changes all the boxes in the "boxes" array from "CenterSize" format to "P1Size" format.
        
        Parameters
        ----------
        boxes : 1-D or 2D list or numpy array of ints or floats
            This contains one or more boxes in "CenterSize" format. In "CenterSize" format each box is represented with array of 4 numbers [cx, cy, w, h] where (cx,cy) and (w,h) represent the center point and size of the box correspondingly.

        Returns
        -------
        same shape and type of the input
            The box(s) in the "P1P2" format. In "P1Size" format each box is represented with array of 4 numbers [x1, y1, w, h] where (x1,y1) and (w,h) represent the top-left corner of the box and size of the box correspondingly.
        """
        
        if type(boxes[0]) in [float, int, np.float32]:
            return np.float32([boxes[0]-boxes[2]/2.0, boxes[1]-boxes[3]/2.0, boxes[2], boxes[3]])

        p1Size = boxes.copy()
        p1Size[:,:2] -= boxes[:,2:]/2.0
        return p1Size

    # ******************************************************************************************************************
    @classmethod
    def getIou(cls, dtBoxes, gtBoxes, crowd=None):
        r"""
        This function returns a matrix of "iou" values for all possible pairs of boxes from `dtBoxes` and `gtBoxes`. If `dtBoxes` has 'd' boxes and `gtBoxes` has 'g' boxes, the returned matrix will be 'd' by 'g'. `dtBoxes` and `gtBoxes` must be in the "P1P2" format (x1, y1, x2, y2).
        
        `crowd` is an array of length g and is True for any gt box that is crowded and False otherwise This class method changes all the boxes in the "boxes" format.
        
        Parameters
        ----------
        dtBoxes : numpy array of ints or floats
            This contains the first set of boxes (Detected Boxes) in "P1P2" format.
            
        gtBoxes : numpy array of ints or floats
            This contains the second set of boxes (Ground-truth Boxes) in "P1P2" format.

        crowd : list, numpy array, or None
            This contains the "crowd" flag value for each one of the boxes in `gtBoxes` (Ground-truth Boxes). For any ground-truth box with `crowd` flag set, we use the area of detected boxes instead of union when calculating the iou.

        Returns
        -------
        numpy array of shape (d,g) and type float32
            The 2d matrix containing the iou value for each pair of detected boxes and ground-truth boxes. The value at i,j position in the matrix contains the iou between the i'th detected box and j'th ground-truth box.

        Note
        ----
        This function works on boxes in "P1P2" format only. In "P1P2" format each box is represented with array of 4 numbers [x1, y1, x2, y2] where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners of the box correspondingly.
        """
        
        x1D, y1D, x2D, y2D = np.split(dtBoxes, 4, axis=1)
        x1G, y1G, x2G, y2G = np.split(gtBoxes, 4, axis=1)

        # Calculate the intersection coordinates (Each one of these are dxg matrixes containing the max/min for every
        # possible pair)
        x1I = np.maximum(x1D, np.transpose(x1G))
        y1I = np.maximum(y1D, np.transpose(y1G))
        x2I = np.minimum(x2D, np.transpose(x2G))
        y2I = np.minimum(y2D, np.transpose(y2G))

        # compute the area of intersection rectangle for every pair (mxn)
#        intersectionArea = np.maximum((x2I - x1I + 1), 0) * np.maximum((y2I - y1I + 1), 0)
        intersectionArea = np.maximum((x2I - x1I), 0) * np.maximum((y2I - y1I), 0)

        # compute the area of both the prediction and ground-truth rectangles
#        dtBoxesArea = (x2D - x1D + 1) * (y2D - y1D + 1)
#        gtBoxesArea = (x2G - x1G + 1) * (y2G - y1G + 1)

        dtBoxesArea = (x2D - x1D ) * (y2D - y1D )
        gtBoxesArea = (x2G - x1G ) * (y2G - y1G )

        unionArea = dtBoxesArea + np.transpose(gtBoxesArea) - intersectionArea
        if crowd is not None:
            g = len(crowd)
            assert g==len(x1G), "Length of 'crowd' (%d) must be the equal to number of GT boxes(%d)!"%(g, len(x1G))
            for j in range(g):
                if crowd[j]:
                    unionArea[:,j] = dtBoxesArea.flatten()
    
        return intersectionArea / unionArea

    # ******************************************************************************************************************
    @classmethod
    def scaleImageAndBoxes(cls, img, boxes=None, res=512, keepAr=True, boxFormat='P1Size'):
        r"""
        This class method scales the specified image to a square `res x res` image. If keepAr is true, the aspect ratio is kept by padding the smaller dimension with zeros (black). It then scales all boxes specified in the `boxes` using the same ratio used to scale the image.
        
        Parameters
        ----------
        img : numpy array
            The image as a numpy array of shape (h,w,3)
            
        boxes : numpy array or None
            A set of bounding boxes for the objects present in the image. The boxes are stored using the format specified by `boxFormat`. It can be `None` or empty which indicates there are no boxes to be scaled.
        
        res : int
            The target resolution to scale to. The returned value is a square `res x res` image.
            
        keepAr : Boolean
            This specifies whether the aspect ratio of the image should be kept when it is scaled.

        boxFormat : str
            This specifies the format of the `boxes`. See the box formats in the Notes section below.
            
        Returns
        -------
            resizedImg : numpy array
                The scaled image as a numpy array of shape (res,res,3)
        
            modifiedBoxes : numpy array or None
                The scaled boxes as a numpy array (Same shape and type as `boxes`) or `None` if `boxes` is None.

            imgSize : tuple
                The size of original image as a 2-tuple (w,h).

        Note
        ----
        The `boxFormat` specifies the format of the boxes. It can be one of the following:
        
        * **P1Size**: The boxes are [x1, y1, w, h] with (x1,y1) and (w,h) as top-left corner and size of the box correspondingly.
        * **CenterSize**: The boxes are [cx, cy, w, h] with (cx,cy) and (w,h) as center point and size of the box correspondingly.
        * **P1P2**: The boxes are [x1, y1, x2, y2] with (x1,y1) and (x2,y2) as top-left and bottom-right corners of the boxes correspondingly.
        """
        
        imgH,imgW,_ = img.shape
        imgSize = (imgW, imgH)
        if keepAr != True:
            scaleX = np.float32(res) / imgW
            scaleY = np.float32(res) / imgH
            resizedImg = cv2.resize(img, (res, res),
                                    interpolation = (cv2.INTER_AREA if scaleX<1.0 else cv2.INTER_CUBIC))
            
            if boxes is None:   return resizedImg, boxes, imgSize
            if len(boxes) == 0: return resizedImg, boxes, imgSize
            modifiedBoxes = boxes * [scaleX, scaleY, scaleX, scaleY]
            return np.float32(resizedImg), modifiedBoxes, imgSize
        
        if imgH>imgW:
            newH, newW = res, imgW * res // imgH
            scale = float(res)/imgH
        else:
            newH, newW = imgH * res // imgW, res
            scale = float(res)/imgW

        # Note: INTER_AREA is best when shrinking and CV_INTER_CUBIC is best when enlarging
        resizedImg = cv2.resize(img, (newW,newH),  interpolation = (cv2.INTER_AREA if scale<1.0 else cv2.INTER_CUBIC))
        # Pad the image to make it square (res x res)
        padTop, padLeft = (res-newH)//2, (res-newW)//2
        resizedImg = np.pad(resizedImg,
                            [(padTop, res-newH-padTop), (padLeft, res-newW-padLeft), (0,0)],
                            'constant')
        
        if boxes is None:   return resizedImg, None, imgSize
        if len(boxes) == 0: return resizedImg, boxes, imgSize
        
        # Now scale and move boxes:
        if boxFormat.lower() in ['p1size', 'centersize']:
            modifiedBoxes = boxes*scale
            modifiedBoxes[:,:2] += [padLeft, padTop]
        else:
            modifiedBoxes = boxes*scale
            modifiedBoxes += [padLeft, padTop, padLeft, padTop]

        return resizedImg, modifiedBoxes, imgSize

    # ******************************************************************************************************************
    @classmethod
    def flipImageHorizontally(cls, img, boxes, boxFormat='P1Size'):
        r"""
        This class method flips an image horizontally. It then moves all the boxes specified in the `boxes` so that they bound the original objects in the flipped images.
        
        Parameters
        ----------
        img : numpy array
            The image as a numpy array of shape (h,w,3)
            
        boxes : numpy array or None
            A set of bounding boxes for the objects present in the image. The boxes are stored using the format specified by `boxFormat`. It can be `None` or empty which indicates there are no boxes to be flipped.
        
        boxFormat : str
            This specifies the format of the `boxes`. See the box formats in the Notes section below.
            
        Returns
        -------
            flippedImg : numpy array
                The flipped image as a numpy array of shape (res,res,3)
        
            flippedBoxes : numpy array or None
                The flipped boxes as a numpy array (Same shape and type as `boxes`) or `None` if `boxes` is None.

        Note
        ----
        The `boxFormat` specifies the format of the boxes. It can be one of the following:
        
        * **P1Size**: The boxes are [x1, y1, w, h] with (x1,y1) and (w,h) as top-left corner and size of the box correspondingly.
        * **CenterSize**: The boxes are [cx, cy, w, h] with (cx,cy) and (w,h) as center point and size of the box correspondingly.
        * **P1P2**: The boxes are [x1, y1, x2, y2] with (x1,y1) and (x2,y2) as top-left and bottom-right corners of the boxes correspondingly.
        """
        
        imgH, imgW, _ = img.shape
        flippedImg = np.flip(img, 1)
        
        if boxes is None:   return flippedImg, boxes
        if len(boxes)==0:   return flippedImg, boxes

        flippedBoxes = boxes.copy()
        if boxFormat.lower() == 'p1size':
            flippedBoxes[:,0] = imgW-boxes[:,0]-boxes[:,2]
        elif boxFormat.lower() == 'centersize':
            flippedBoxes[:,0] = imgW-boxes[:,0]
        else:
            flippedBoxes[:,0] = imgW-boxes[:,2]
            flippedBoxes[:,2] = imgW-boxes[:,0]

        return flippedImg, flippedBoxes

    # ******************************************************************************************************************
    @classmethod
    def zoomOutAndMove(cls, img, boxes, newRes, res, offset=(0,0), boxFormat='P1Size'):
        r"""
        This class method scales the image down and then move the smaller image by an offset specified by `offset`. The return value is a black `res x res` image that contains a `newRes x newRes` image at a location that is specified by `offset` (from the center of the image).

        It then scales and moves all the boxes specified in the `boxes` so that they bound the original objects in the resized/moved images.
        
        Parameters
        ----------
        img : numpy array
            The image as a numpy array of shape (h,w,3)
            
        boxes : numpy array or None
            A set of bounding boxes for the objects present in the image. The boxes are stored using the format specified by `boxFormat`. It can be `None` or empty which indicates there are no boxes to be scaled/moved.
        
        newRes : int
            The new resolution of the image. The returned image contains a `newRes x newRes` image.
            
        res : int
            The resolution of the returned image.
        
        offset : tuple
            This 2-tuple specifies the offset of the scaled image from the center point of the image.
        
        boxFormat : str
            This specifies the format of the `boxes`. See the box formats in the Notes section below.
            
        Returns
        -------
            movedImg : numpy array
                The scaled/moved image as a numpy array of shape (res,res,3)
        
            movedBoxes : numpy array or None
                The scaled/moved boxes as a numpy array (Same shape and type as `boxes`) or `None` if `boxes` is None.

        Note
        ----
        The `boxFormat` specifies the format of the boxes. It can be one of the following:
        
        * **P1Size**: The boxes are [x1, y1, w, h] with (x1,y1) and (w,h) as top-left corner and size of the box correspondingly.
        * **CenterSize**: The boxes are [cx, cy, w, h] with (cx,cy) and (w,h) as center point and size of the box correspondingly.
        * **P1P2**: The boxes are [x1, y1, x2, y2] with (x1,y1) and (x2,y2) as top-left and bottom-right corners of the boxes correspondingly.
        """
        resizedImg = cv2.resize(img, (newRes,newRes),  interpolation =cv2.INTER_AREA)
        movedImg = np.pad(resizedImg, [(offset[1],res-newRes-offset[1]), (offset[0], res-newRes-offset[0]), (0,0)], 'constant')

        if boxes is None:   return movedImg, boxes
        if len(boxes)==0:   return movedImg, boxes

        movedBoxes = boxes.copy()
        if boxFormat.lower() in ['p1size', 'centersize']:
            movedBoxes = movedBoxes*np.float32(newRes)/res
            movedBoxes[:,:2] += [offset[0], offset[1]]
        else:
            movedBoxes = movedBoxes*np.float32(newRes)/res
            flippedBoxes = [offset[0], offset[1], offset[0], offset[1]]
        
        return movedImg, movedBoxes

    # **********************************************************************************************************************
    def setAcnchorBoxes(self, anchorBoxes):
        r"""
        Sets the anchor boxes for this dataset. The anchor boxes are created by an SSD model and passed to this class so that they can be used during training and evaluation of the models.
        
        Parameters
        ----------
        anchorBoxes : numpy array
            An nx4 numpy array, where n is the number of anchor boxes. The boxes are in "CenterSize" format [cx, cy, w, h] with all values normalized to an image size of `1.0 x 1.0`.
        """
        
        self.anchorBoxes = anchorBoxes
        self.anchorsBoxesP1P2 = CocoDSet.centerSizeToP1P2(self.anchorBoxes)

    # **********************************************************************************************************************
    def getGroundTruth(self, labels, boxes):
        r"""
        This function receives lists of objects and their locations on an image and creates the ground-truth information used for training a model. The ground-truth information includes the label and location information for each one of the anchor boxes defined by the model.
        
        Parameters
        ----------
        labels : numpy array
            This 1-D array contains the class of each object in an image.
        
        boxes : numpy array
            This 2-D matrix contains the box information for each object in an image. The boxes are in "P1Size" format. The number of boxes in this array should match the number of labels in the `labels` parameter.
        
        Returns
        -------
        gtLabels : numpy array
            The label for each anchor box. The number of items in the array is equal to the number of anchor boxes defined by the model. (See setAcnchorBoxes)
            
        gtBoxAdj : numpy array
            This the adjustment applied to each anchor box to match it to one of the ground-truth boxes in the image. Shape: (numAnchors, 4)
            
        gtMask : numpy array
            A Foreground/background indicator for each anchor box: 1->Foreground, -1->Background, 0->Neutral
            
        gtIous : numpy array
            This is a 1-D array of IOU values for each anchor box. For the i'th anchor box, this function finds the box in the `boxes` array that has the highest IOU with the anchor box. It then sets gtIous[i] to this maximum value.
        """

        numAnchors = len(self.anchorsBoxesP1P2)
        numBoxes = len(boxes)
        if numBoxes == 0:
            # We should mark everything as background:
            gtBoxes = self.anchorsBoxesP1P2                     # Same as anchor boxes (0 adjustment)
            gtLabels = np.zeros( numAnchors, dtype=np.int32)    # All 0's for background label
            gtMask = -np.ones( numAnchors, dtype=np.int32)      # All -1's for background
            gtIous = np.zeros( numAnchors, dtype=np.int32)      # All 0's for IOUs
            
        else:
            normBoxP1P2 = CocoDSet.p1SizeToP1P2(boxes)/self.resolution  # Convert to P1P2 and normalize boxes
            ious = CocoDSet.getIou(self.anchorsBoxesP1P2, normBoxP1P2)  # This is a numAnchors x numBoxes matrix
            maxIndexes = np.argmax( ious, axis=1 )  # This is an array of values Xi with length "numAnchors"
                                                    # where Xi is the index of a box in "boxes" that has the highest IOU with
                                                    # i'th anchor box (0<Xi<numBoxes)

            maxIous = ious[ range(numAnchors), maxIndexes]  # The max IOU for each anchor box. Shape: (numAnchors,)
            
            gtLabels = labels[ maxIndexes ] # Ground-truth Labels. The i'th element Li is the label corresponding to the
                                            # box in boxes that has the highest IOU with i'th anchor box. Shape: (numAnchors,)
                                            
            gtBoxes = normBoxP1P2[ maxIndexes, : ]  # Ground-truth boxes. The i'th element Bi is the box in boxes
                                                    # (Normalized P1P2) that has the highest IOU with i'th anchor
                                                    # box. Shape: (numAnchors, 4)
            gtIous = maxIous
            
            # Hard Negative Mining:
            hardNegativeFg = 0.5    # The threshold for foreground boxes
            hardNegativeBg = 0.5    # The threshold for background boxes
            fgIndexes = np.where(maxIous > hardNegativeFg)[0]   # Indexes of anchor boxes with high iou with one of the boxes
            bgIndexes = np.where(maxIous < hardNegativeBg)[0]   # Indexes of anchor boxes that don't have high iou with any box
            gtLabels[ bgIndexes ] = 0   # Set the label to 0 (background) for anchor boxes that don't have high iou with any box
            gtIous[ bgIndexes ] = 0     # Set the iou to 0 (background) for anchor boxes that don't have high iou with any box

            gtMask = np.zeros( numAnchors, dtype=np.int32)
            gtMask[ fgIndexes ] = 1     # foreground
            gtMask[ bgIndexes ] = -1    # Background
            
            # Now the other direction
            maxIndexes = np.argmax( ious, axis=0 )  # This is an array of values Xj with length "numBoxes"
                                                    # where Xj is the index of an anchor box that has the highest IOU with
                                                    # j'th box in "boxes" (0<Xj<numAnchors)
                                                    
            gtLabels[ maxIndexes ] = labels         # For each anchor box that has highest IOU with i'th box, set the ground
                                                    # Truth label to i'th label
            
            gtBoxes[ maxIndexes ] = normBoxP1P2     # For each anchor box that has highest IOU with i'th box, set the ground
                                                    # Truth box to the i'th box (Normalized P1P2)
                                                    
            gtMask[ maxIndexes ] = 1                # For each anchor box that has highest IOU with i'th box, make it a
                                                    # foreground box
                                                    
            gtIous[ maxIndexes ] = ious[ maxIndexes, range(numBoxes) ]

        # For training, we need the adjustment values to center point (cx,cy) and box size (w,h) for each anchor box.
        anchorCenters, anchorSizes = self.anchorBoxes[:,:2], self.anchorBoxes[:,2:]

        p1, p2 = gtBoxes[:,:2], gtBoxes[:,2:]
        gtBoxCenters = (p1+p2)/2.
        gtBoxSizes = (p2-p1)
        assert np.all(gtBoxSizes>0)
        
        centerVar, sizeVar = 0.1, 0.2   # Variance values
        centerAdj = ( gtBoxCenters - anchorCenters)/(anchorSizes * centerVar)   # Center point adjustments for all ground-truth boxes
        sizeAdj = np.log( gtBoxSizes/anchorSizes )/sizeVar                      # Size adjustments for all ground-truth boxes
        gtBoxAdj = np.concatenate((centerAdj, sizeAdj), axis=1)                 # Ground-truth Box adjustments
        return gtLabels, gtBoxAdj, gtMask, gtIous

    # MARK: ------- Image and Bounding Box Display functions -------
    # ******************************************************************************************************************
    @classmethod
    def showImageAndBoxes(cls, image, boxes, labels, title=None):
        r"""
        This class method shows the specified image and the bounding boxes in a matplotlib.pyplot diagram.
        
        Parameters
        ----------
        image : numpy array
            The image as a numpy array of shape (height, width, 3) in BGR format.
            
        boxes : numpy array or list
            Each item in `boxes` represents a single box in "P1Size" format [x,y,w,h].
            
        labels : numpy array or list
            The class of each object in the image. This is used to look up the class name and show it as caption for the object on the image. Th number of labels should match the number of boxes.
            
        title : str
            The title used for the displayed image.
        
        Note
        ----
        This function blocks current thread until the user closes the image window manually.
        """
        
        global plt
        if plt is None: import matplotlib.pyplot as plt  # Lazy import!
                    
        img = (image/256.0)[..., ::-1]    # Convert to RGB
        imgH,imgW,_ = img.shape

        currentAxis = plt.gca()
        colors = plt.cm.hsv(np.linspace(0, 1, 121)).tolist()
        for i, box in enumerate(boxes):
            caption = CocoDSet.classNames[ labels[i] ]
            x,y, w,h = box
                
            currentAxis.add_patch(plt.Rectangle((x,y), w,h, fill=False, edgecolor=colors[labels[i]], linewidth=2))
            currentAxis.text(x,y, caption, bbox={'facecolor':colors[labels[i]]},
                             verticalalignment='top')

        imgplot = plt.imshow(img)
        if title is not None:   plt.title(title)
        plt.show()

    # ******************************************************************************************************************
    def showSample(self, sampleIndex=None, sampleId=None, title=None):
        r"""
        This function shows one of the samples in this dataset as specified by the `sampleIndex`
        or `sampleId`.

        Parameters
        ----------
        sampleIndex : int or None
            The sample index. This index is used to get the specified sample in the dataset.
            
        sampleId : int or None
            The sample identifier. This is used to find the sample index. The sample index can then be used get the specified sample in the dataset.
                        
        title : str
            The title used for the displayed image.
        
        Note
        ----
            - If `sampleIndex` is specified `sampleId` is ignored, otherwise `sampleId` must be specified.
            - This function blocks current thread until the user closes the image window manually.
        """
        
        if sampleIndex is None:
            assert sampleId is not None, "Eighter 'sampleIndex' or 'sampleId' must be specified!"
            sampleIndex = self.imgIdToIndex[sampleId]
                                         
        sampleId, classes, boxes, _, _ = self.samples[sampleIndex]
        img = self.getImage(sampleId)
        CocoDSet.showImageAndBoxes(img, boxes, classes, title)

    # ******************************************************************************************************************
    @classmethod
    def showInferResults(cls, image, boxes=[], labels=[], scores=[], arKept=True, title=None):
        r"""
        This function shows the results of inference. First the `image` is sent to the model to detect all the objects in the image. The detected information include bounding boxes, the labels, and scores (or confidence factors) for each detected object. The information is then passed to this function to display the image together with the detected objects.

        Parameters
        ----------
        image : numpy array
            The image as a numpy array of shape (height, width, 3) in BGR format.

        boxes : numpy array or list
            The detected boxes in P1Size format with normalized (between 0 and 1) coordinates. The number of labels, scores, and boxes, should match.

        labels : numpy array or list
            The class of each detected object in the image. This is used to look up the class name and show it as caption for each detected object in the image. The number of labels, scores, and boxes, should match.

        scores : numpy array or list
            The score (or confidence) for each detected object in the image. The number of labels, scores, and boxes, should match.

        arKept : Boolean
            If True, it means the aspect ratio of the image was kept when it was fed to the model for inference.
            
        title : str
            The title used for the displayed image.
        
        Note
        ----
        This function blocks current thread until the user closes the image window manually.
        """
        
        global plt
        if plt is None: import matplotlib.pyplot as plt  # Lazy import!
        
        img = (image/256)[..., ::-1]    # Convert to RGB
        imgH,imgW,_ = img.shape

        currentAxis = plt.gca()
        colors = plt.cm.hsv(np.linspace(0, 1, 121)).tolist()
        for i, box in enumerate(boxes):
            caption = "%s: %2.1f%%"%(CocoDSet.classNames[ labels[i] ], scores[i]*100)
            if arKept:
                if imgW < imgH:
                    box *= imgH
                    box[0] -= (imgW*512/imgH)/2
                else:
                    box *= imgW
                    box[1] -= (imgH*512/imgW)/2
            else:       # Forced resize. Aspect Ratio was lost.
                box *= [imgW,imgH,imgW,imgH]
            x,y, w,h = np.int32( np.rint(box) )
                
            currentAxis.add_patch(plt.Rectangle((x,y), w,h, fill=False, edgecolor=colors[ labels[i] ], linewidth=2))
            currentAxis.text(x,y, caption, bbox={'facecolor':colors[ labels[i] ]},
                             verticalalignment='top')

        imgplot = plt.imshow(img)
        if title is not None:   plt.title(title)
        plt.show()

    # MARK: ---------------- Sample/Batch Iterations ----------------
    # ******************************************************************************************************************
    def randomMutate(self, img, boxes):
        r"""
        This function randomly mutates the specified image. It can be used for data augmentation to improve the training of the model.
        
        Currently 2 types of mutations are supported:
        
            * Horizontal Flip (%25 probability)
            * Zoom out and move (%25 probability)
        
        Also %50 of the time the original image is returned without any modifications. Please refer to `flipImageHorizontally` and `zoomOutAndMove` functions for more detail about these mutations.
        
        Parameters
        ----------
        img : numpy array
            The image as a numpy array of shape (height, width, 3) in BGR format.

        boxes : numpy array or list
            Each item in `boxes` represents a single box in "P1Size" format [x,y,w,h].
            
        Returns
        -------
            mutatedImg : numpy array
                The modified image as a numpy array (same shape and format as the original image)
        
            mutatedBoxes : numpy array
                The modified boxes as a numpy array (Same shape and type as `boxes`).
        """

        # TODO: Make unit test for this function
        randNum = np.random.random()
        
        # %50 of the time just return the original image
        if randNum < .50:   return  img, boxes
        
        # %25 of the time just return the original image
        if randNum < .75:   return  CocoDSet.flipImageHorizontally(img, boxes)
        
        # and %25 of the time return the zoomed out image
        scaledSize = np.random.randint(int(.5*self.resolution),
                                       int(.9*self.resolution)) # scale by a factor of .5 to .9 of the image size
        maxMove = (self.resolution-scaledSize)//2               # Making sure all boxes remain in frame after move
        move = (np.random.randint(0,maxMove), np.random.randint(0,maxMove))
        return  CocoDSet.zoomOutAndMove(img, boxes, scaledSize, self.resolution, move)
        
    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        r"""
        This method returns a batch of samples and labels from the dataset as specified by the list of indexes in the `batchIndexes` parameter.
        
        If the anchor boxes are already given to this dataset (See `setAcnchorBoxes`) then it is assumed we are training a model. In this case this function provides the pre-processed image together with ground-truth information (see `getGroundTruth`).
        
        For the training case, you can enable data augmentation using the `augment` property. By default this is set to False.
        
        If the anchor boxes are not available, it is assumed we are in inference mode. In this case the pre-processed images are returned together with a list of tuples (imageId, imageSize).
        
        Parameters
        ----------
        batchIndexes : list of int
            A list of indexes used to pick samples from the dataset.
            
        Returns
        -------
        images : numpy array
            The batch images specified by the `batchIndexes`. Each image is resized, pre-processed, and possibly mutated and returned as a numpy array.
            
        labels : list of 2-tuples or a 3-tuple of numpy arrays
        
            * If training, a 3-tuple is used as "labels" for the batch. The tuple contains the following items:
            
                * Ground-truth labels
                * Ground-truth box adjustments
                * Ground-truth background masks
            
            * If inferring, a list of tuples (imageId, imageSize).
        """

        images = []
        labels = []
        gtBatchLabels = []
        gtBatchBoxAdj = []
        gtBatchMask = []
        for i in batchIndexes:
            imgId, classesInImg, boxesInImg, _, _ = self.samples[ i ]
            if len(boxesInImg)>0:
                boxSizes = boxesInImg[:,2:]
                if np.all(boxSizes>0) == False:
                    print("ImgId: %d", imgId)
                    print("Boxes:\n", boxesInImg)
                    exit(0)
            
            scaledImg, scaledBoxes, imgSize = CocoDSet.scaleImageAndBoxes(self.getImage(imgId), boxesInImg,
                                                                          self.resolution, self.keepAr)

            if self.augment:  # Data Augmentation:
                scaledImg, scaledBoxes = self.randomMutate(scaledImg, scaledBoxes)
                
            images += [ scaledImg ]
            if self.anchorBoxes is None:    labels += [ (imgId, imgSize) ]
            else:
                gtLabels, gtBoxAdj, gtMask, gtIous = self.getGroundTruth(classesInImg, scaledBoxes)
                gtBatchLabels += [ gtLabels ]
                gtBatchBoxAdj += [ gtBoxAdj ]
                gtBatchMask += [ gtMask ]

        if self.anchorBoxes is None:    return (images-CocoDSet.bgrMeans), labels
        return (images-CocoDSet.bgrMeans), (np.int32(gtBatchLabels), np.float32(gtBatchBoxAdj), np.int32(gtBatchMask))
  

    # MARK: ------------------ Evaluation Functions ------------------
    # ******************************************************************************************************************
    def calculateIous(self, results, quiet=False):
        r"""
        This receives the inference results and calculates all IOU values for all pairs of bounding boxes between detected objects and the ground-truth boxes.
        
        Parameters
        ----------
        results : list of tuples
            The results is a list of tuples [ (IMG0, DT0), (IMG1, DT1), ... ] where:
            
                * IMGi : The i'th image id.
                * DTi : The predicted information for the i'th image. It is a 3-tuple containing numpy arrays: ([CLASS0, CLASS1, ...], [BOX0, BOX1, ...], [SCORE0, SCORE1, ...])
                
                    * CLASSj : The class index (between 1 and numClasses - 0/background should not appear here).
                    * BOXj : The predicted bounding box for the j'th object detected in the i'th image in P1P2 format.
                    * SCOREj : The score (or confidence) for the j'th detected object.
                
            The classes, boxes, and scores are sorted based on the descending order of scores. (The best predictions appear first)
        
        quiet : Boolean
            If true, this function shows the progress by printing current image id and the number of processed images.
            
        Returns
        -------
        ious : 2-D list of numpy arrays.
            IOU values for all pairs of bounding boxes between detected objects and the ground-truth objects. The ``ious[ i ][ c ]`` can be:
            
            * None: This means the image `i`, does not have class `c` in its ground truth objects or class `c` was not detected by the Model.
            * A DxG matrix: (`D`: number of detect boxes, and `G`: number of ground truth boxes in the `i`'th image with class `c`). The (d,g)'th element of the matrix contains the IOU value between the `d`'th box in detected objects and `g`'th box in the ground-truth objects for class `c` in image `i`.
        """

        numClasses = len(CocoDSet.classNames)
        numImages = len(results)
        startTime = time.time()
        ious = []
        for img, (imgId, dt) in enumerate(results):
            if quiet==False:
                myPrint( '\r  Calculating IoUs - Image ID:%-6d (%d of %d) ... '%(imgId, img+1, numImages), False)
            dtClasses, dtBoxes, dtScores = dt
            imageIous = [ None ]
            _, gtClasses, gtBoxes, gtAreas, gtIsCrowds = self.samples[ self.imgIdToIndex[ imgId ] ]
            gtBoxes = CocoDSet.p1SizeToP1P2(gtBoxes)
            for classId in range(1,numClasses): # Only real classes not the "background"
                gtIndexes = np.where( (gtClasses==classId) )[0]
                dtIndexes = np.where( (dtClasses==classId) )[0]
                if (len(gtIndexes) == 0) or (len(dtIndexes) == 0):
                    # We need at least one ground-truth and one detected object for this class
                    iou = None
                else:
                    iou = CocoDSet.getIou(dtBoxes[ dtIndexes ], gtBoxes[ gtIndexes ], gtIsCrowds[ gtIndexes ])
                imageIous += [ iou ]

            ious += [ imageIous ]
        if quiet==False:
            myPrint('\r  Calculating IoUs - Done (%.1f Seconds)                       '%(time.time()-startTime))
        return ious

    # ******************************************************************************************************************
    def findMatches(self, results, ious, iouMins, areaAndMaxDetInfo, quiet=False):
        r"""
        This function matches the ground-truth boxes to the detected boxes based on confidence and IOU values.
        
        Parameters
        ----------
        results : list of tuples
            The results is a list of tuples [ (IMG0, DT0), (IMG1, DT1), ... ] where:
            
            * IMGi : The i'th image id.
            * DTi : The predicted information for the i'th image. It is a 3-tuple containing numpy arrays: ([CLASS0, CLASS1, ...], [BOX0, BOX1, ...], [SCORE0, SCORE1, ...])
            
                * CLASSj : The class index (between 1 and numClasses - 0/background should not appear here).
                * BOXj : The predicted bounding box for the j'th object detected in the i'th image in P1P2 format.
                * SCOREj : The score (or confidence) for the j'th detected object.
                
            The classes, boxes, and scores are sorted based on the descending order of scores. (The best predictions appear first)
        
        ious : 2-D list of numpy arrays.
            IOU values for all pairs of bounding boxes between detected objects and the ground objects. The ``ious[ i ][ c ]`` can be:
            
            * None: This means the image `i`, does not have class `c` in its ground truth objects or class `c` was not detected by the Model.
            * A DxG matrix: (`D`: number of detect boxes, and `G`: number of ground truth boxes in the `i`'th image with class `c`). The (d,g)'th element of the matrix contains the IOU value between the `d`'th box in detected objects and `g`'th box in the ground-truth objects for class `c` in image `i`.
                
        iouMins : list
            This is a list of IOU threshold values to be used when finding the matches. Different sets of matches are found for each IOU threshold value. Typically this list contains the threshold values from 0.50 to 0.95 in steps of 0.05.
            
        areaAndMaxDetInfo : A list of tuples
            This is a list of 4-tuples (areaMin, areaMax, maxDet, descStr) where:
            
            * areaMin : Minimum area for the boxes to consider. If a detected or ground-truth box has an area smaller than this value it is not considered to match with other boxes.
            * areaMax : Maximum area for the boxes to consider. If a detected or ground-truth box has an area larger than this value it is not considered to match with other boxes.
            * maxDet : The maximum number of detected boxes to consider when matching detected boxes to the ground-truth boxes. Since the boxes are sorted, the selected boxes are the ones with highest confidence value.
            * descStr : A text string explaining the combination of parameters.
            
        quiet : Boolean
            If true, this function shows the progress by printing current image id and the number of processed images.
            
        Returns
        -------
        numGts : 2-D list
            ``numGts[c][a]`` is the number of ground-truth items in the matches for class `c` and config area/maxDet combination `a`.
        
        dtScores : numpy array
            Confidence of a match for all combinations. ``dtScores[c][a][i][d]`` is the score (confidence) of class `c` for `d`'th detected object with `a`'th area/maxDet combination and `i`'th value in `iouMins`
            
        dtMatches:
            The match flags for all combinations and pairs of objects. ``dtMatches[c][a][i][d]`` is the match indicator for `d`'th detected if it is considered in class `c` with `a`'th area/maxDet combination and `i`'th `iouMin`. The match indicator is one of the following:
            
            * -1: There is no match. No ground-truth item could be matched to the detected object.
            * 0: Ignore this detected object. (The area did not match the `a`'th area/maxDet combination)
            * 1: There is a match. At least one ground-truth object was matched to `d`'th detected object.
        """

        startTime = time.time()

        numClasses = len(CocoDSet.classNames)
        numImages = len(results)
        A = len(areaAndMaxDetInfo)                  # Number of area/maxDet combinations
        I = len(iouMins)                            # Number of IoU Threshold values
        
        dtScores = [[] for _ in range(numClasses)]  # For each class this is a numpy array of type np.float32 and shape
                                                    # (A,I,D). Each element can be:
                                                    #   -1: The detected object should be ignored (based on Area/maxDet)
                                                    #       When the scores are sorted later, this causes all ignored
                                                    #       detections to go to the end of the list.
                                                    #   >=0: The score of each detected object.
        dtMatches = [[] for _ in range(numClasses)] # For each class this is a numpy array of type np.int32 and shape
                                                    # (A,I,D). Each element can be one of the following:
                                                    #   1: The detected object was matched to a ground-truth (True
                                                    #      Positive)
                                                    #   0: The detected object should be ignored (based on Area/maxDet)
                                                    #  -1: The detected object was not matched to any ground-truth
                                                    #      (False Positive)
        numGts = [np.zeros(A, dtype=np.float32) for _ in range(numClasses)] # For each class this is a numpy array of
                                                                            # type np.float32 and shape (A,). Each element
                                                                            # is the number of ground-truth objects for
                                                                            # the specified area size and maxDet

        maxMaxDet = max([maxDet for _, _, maxDet, _ in areaAndMaxDetInfo])
        
        for img, (imgId, dt) in enumerate(results):
            if quiet==False:
                myPrint( '\r  Finding matches - Image ID:%-6d (%d of %d) ... '%(imgId, img+1, numImages), False)
            dtClasses4Img, dtBoxes4Img, dtScores4Img = dt

            _, gtClasses4Img, _, gtAreas4Img, gtIsCrowds4Img = self.samples[ self.imgIdToIndex[ imgId ] ]

            for classId in range(1,numClasses): # Only real classes not the "background"
                gtIndexes = np.where( (gtClasses4Img==classId) )[0][:maxMaxDet]
                dtIndexes = np.where( (dtClasses4Img==classId) )[0]
                D = len(dtIndexes)
                G = len(gtIndexes)
                if D == 0 and G == 0: continue

                gtIsCrowds = gtIsCrowds4Img[ gtIndexes ]
                gtAreas = gtAreas4Img[ gtIndexes ]
                
                classMatches = [None]*A
                classScores = [None]*A
                classNumGt = [0]*A
                
                dtAreas = None  # Lazy calculation!
                scr = dtScores4Img[dtIndexes].tolist()
                for a, (areaMin, areaMax, maxDet, _) in enumerate(areaAndMaxDetInfo):
                    if G>0:
                        # We want to ignore the ground-truth boxes if their area is not in the range or gtIsCrowds is False
                        keepGt = (gtAreas>=areaMin) * (gtAreas<=areaMax) * (gtIsCrowds==False)
                        classNumGt[a] = keepGt.sum()
                        keepGt = keepGt.tolist()
                    
                    # Ignore detections after 'maxDet':
                    if ious[img][classId] is None:
                        keepDt = []
                        if D>0:
                            # Count an un-matched DT as a failure only if its area is in the specified range
                            if dtAreas is None:
                                dtBoxes = dtBoxes4Img[dtIndexes,:]
                                dtAreas = (dtBoxes[:,2]-dtBoxes[:,0])*(dtBoxes[:,3]-dtBoxes[:,1])
                            keepDt = ((dtAreas>=areaMin) * (dtAreas<=areaMax)).tolist()
                        
                        matches = [[-1     if (d<maxDet and keepDt[d]) else 0  for d in range(D)] for _ in range(I)]
                        scores =  [[scr[d] if (d<maxDet and keepDt[d]) else -1 for d in range(D)] for _ in range(I)]
                    else:
                        imgClsIous = ious[img][classId].tolist()
                        numDt = min(D, maxDet)
                        
                        matches = [[-1     if d<maxDet else 0  for d in range(D)] for _ in range(I)]
                        scores =  [[scr[d] if d<maxDet else -1 for d in range(D)] for _ in range(I)]
                        for i,iouMin in enumerate(iouMins):
                            gtTaken = [False]*G
                            for d in range(numDt):
                                curMin = iouMin
                                bestG = -1
                                for g in range(G):
                                    if gtTaken[g]:                  continue
                                    if keepGt[g]==False:            continue
                                    if imgClsIous[d][g] < curMin:   continue
                                    bestG = g
                                    curMin = imgClsIous[d][g]
                                if bestG<0:
                                    for g in range(G):
                                        if gtTaken[g]:
                                            if gtIsCrowds[g]==False:
                                                continue
                                        if imgClsIous[d][g] < curMin:   continue
                                        bestG = g
                                        curMin = imgClsIous[d][g]
                                
                                if bestG<0:
                                    # Count an un-matched DT as a failure only if its area is in the specified range
                                    if dtAreas is None:
                                        dtBoxes = dtBoxes4Img[dtIndexes,:]
                                        dtAreas = (dtBoxes[:,2]-dtBoxes[:,0])*(dtBoxes[:,3]-dtBoxes[:,1])
                                    if (dtAreas[d]>=areaMin and dtAreas[d]<=areaMax): continue
                                    # Otherwise ignore this DT
                                    matches[i][d] = 0
                                    scores[i][d] = -1
                                else:
                                    gtTaken[bestG]=True
                                    if keepGt[bestG]:
                                        matches[i][d] = 1
                                    else:
                                        matches[i][d] = 0
                                        scores[i][d] = -1
                    classMatches[a] = matches
                    classScores[a] = scores

                dtMatches[ classId ] += [ np.int32(classMatches) ]      # classMatches is A x I x D
                dtScores[ classId ] += [ np.float32(classScores) ]      # classScores is A x I x D
                numGts[ classId ] += np.float32(classNumGt)
                
        for classId in range(1,numClasses): # Only real classes not the "background"
            if len(dtMatches[classId]) > 0:
                dtMatches[classId] = np.concatenate(dtMatches[classId], -1)     # [ A x I x D ]
                dtScores[classId] = np.concatenate(dtScores[classId], -1)       # [ A x I x D ]
        
        if quiet==False:
            myPrint('\r  Finding matches - Done (%.1f Seconds)                     '%(time.time()-startTime))
        return numGts, dtScores, dtMatches
        
    # ******************************************************************************************************************
    def calStats(self, numGts, dtScores, dtMatches, quiet=False):
        r"""
        Calculates and returns all the statistics for the evaluation of inference results.
        
        Parameters
        ----------
        numGts : 2-D list
            ``numGts[c][a]`` is the number of ground-truth items in the matches for class `c` and config area/maxDet combination `a`.
        
        dtScores : numpy array
            Confidence of a match for all combinations. ``dtScores[c][a][i][d]`` is the score (confidence) of class `c` for `d`'th detected object with `a`'th area/maxDet combination and `i`'th value in `iouMins`
            
        dtMatches:
            The match flags for all combinations and pairs of objects. ``dtMatches[c][a][i][d]`` is the match indicator for `d`'th detected object if it is considered in class `c` with `a`'th area/maxDet combination and `i`'th `iouMin`. The match indicator is one of the following:
            
            * -1: There is no match. No ground-truth item could be matched to the detected object.
            * 0: Ignore this detected object. (The area did not match the `a`'th area/maxDet combination)
            * 1: There is a match. At least one ground-truth object was matched to `d`'th detected object.

        quiet : Boolean
            If true, this function shows the progress by printing current class and the number of processed classes.
            
        Returns
        -------
        ap50 : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet calculated with IOU threshold of .50.

        ap75 : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet calculated with IOU threshold of .75.

        ap : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet averaged over IOU threshold values 0.50, 0.55, ... 0.95.
            
        ar : numpy array
            A numpy array containing the average recall for all combinations of area/maxDet averaged over IOU threshold values 0.50, 0.55, ... 0.95.
        """

        numClasses = len(CocoDSet.classNames)

        N = 101
        recallTicks = np.arange(N)/100.0
        
        A, I, _ = dtMatches[1].shape
        sumAp50 = np.zeros(A)
        sumAp75 = np.zeros(A)
        sumAp = np.zeros(A)
        sumAr = np.zeros(A)
        C = np.zeros(A)
        
        startTime = time.time()
        for classId in range(1,numClasses): # Only real classes not the "background"
            if quiet==False:
                myPrint( '\r  Processing the matches - Class %-2d ... '%(classId), False)
            classMatches = dtMatches[classId]   # [ A x I x D ]
            classScores = dtScores[classId]     # [ A x I x D ]
            classNumGts = numGts[classId]       # [ A ]

            C += (classNumGts>0)*1.0
            if np.all(classNumGts==0):      continue
            if len(classMatches) == 0:      continue
            if classMatches.shape[2] == 0:
                if quiet==False:
                    print("Warning: Class %d (%s): No objected detected in all images!"%(classId,CocoDSet.classNames[classId]))
                continue
            
            # Sort the classMatches and classScores based on the class scores. The unmatched objects have a score of
            # -1 and will appear at the end of the list. Better matches (higher scores) appear at the beginning of the
            # list.
            dtScoreIndexes = np.argsort(-classScores, kind='mergesort') # [ A x I x D ]
            
            aIndexes = np.arange(A).reshape( A, 1, 1 )
            iIndexes = np.arange(I).reshape( 1, I, 1 )
            classMatches = classMatches[aIndexes, iIndexes, dtScoreIndexes]                     # [ A x I x D ]
            classScores = classScores[aIndexes, iIndexes, dtScoreIndexes]                       # [ A x I x D ]
            
            tp = (classMatches==1)
            tpCum = np.cumsum( tp, -1, np.float32)                      # [ A x I x D ]   Increased when we have a TP
            numPredCum = np.cumsum( (classMatches!=0), -1, np.float32)  # [ A x I x D ]   Increased when we have a pred
            
            precision = np.divide(tpCum, numPredCum, out=np.zeros_like(tpCum), where=numPredCum!=0) # [ A x I x D ]
            classNumGts = classNumGts.reshape(A, 1, 1)
            recall = np.divide(tpCum, classNumGts, out=np.zeros_like(tpCum), where=classNumGts!=0)  # [ A x I x D ]

            # Make precision monotonically decreasing
            precision = np.maximum.accumulate(precision[:,:,::-1], axis=-1)[:,:,::-1]   # [ A x I x D ]

            def getRecallTickIndexes(a):
                return np.searchsorted(a, recallTicks, side='left')
            pIndexes = np.apply_along_axis(getRecallTickIndexes, -1, recall)

            precision = np.concatenate([precision, np.zeros((A,I,1))],-1)   # [ A x I x (D+1) ]

            aIndexes = np.arange(A).reshape( A, 1, 1 )
            iIndexes = np.arange(I).reshape( 1, I, 1 )
            precision = precision[aIndexes, iIndexes, pIndexes]         # [A x I x N]   (N: Number of Ticks = 101)

            recall = recall[:,:,-1]                                     # [A x I]
            
            sumAp50 += precision[:,0,:].sum(-1).flatten()   # [A]   sum of all precisions at a=0 iouMin= 0.5
            sumAp75 += precision[:,5,:].sum(-1).flatten()   # [A]   sum of all precisions at a=5 iouMin= 0.75
            sumAp += precision.sum((1,2))                   # [A]   sum of all precisions at all iouMins (0.50 .. 0.95)
            sumAr += recall.sum(-1)                         # [A]   sum of all recalls at all iouMins (0.50 .. 0.95)
            
        if quiet==False:
            myPrint('\r  Processing the matches - Done (%.1f Seconds)                    '%(time.time()-startTime))
        return sumAp50/(C*N), sumAp75/(C*N), sumAp/(C*I*N), sumAr/(C*I)
    
    # ******************************************************************************************************************
    def evaluate(self, results, isTraining=False, quiet=False):
        r"""
        This function returns the results of inference and evaluates the results. It then returns the result statistics together with a human readable text string containing details of evaluation results in the form of a table.
        
        Parameters
        ----------
        results : list of tuples
            The results is a list of tuples [ (IMG0, DT0), (IMG1, DT1), ... ] where:
            
            * IMGi : The i'th image id.
            * DTi : The predicted information for the i'th image. It is a 3-tuple containing numpy arrays: ([CLASS0, CLASS1, ...], [BOX0, BOX1, ...], [SCORE0, SCORE1, ...])
            
                * CLASSj : The class index (between 1 and numClasses - 0/background should not appear here).
                * BOXj : The predicted bounding box for the j'th object detected in the i'th image in P1P2 format.
                * SCOREj : The score (or confidence) for the j'th detected object.
              
            The classes, boxes, and scores are sorted based on the descending order of scores. (The best predictions appear first)

        isTraining : Boolean
            True means this function is called during the training (at the end of each epoch) to show the intermediate results during the training. In this case the results are calculated only for a single combination of area/maxDet.
            
        quiet : Boolean
            If true, this function shows the progress during the calculations and prints the details of results in the form of a table. Otherwise, this function does not print anything during the process.

        Returns
        -------
        ap50 : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet calculated with IOU threshold of .50.

        ap75 : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet calculated with IOU threshold of .75.

        ap : numpy array
            A numpy array containing the average precision for all combinations of area/maxDet averaged over IOU threshold values 0.50, 0.55, ... 0.95.
            
        ar : numpy array
            A numpy array containing the average recall for all combinations of area/maxDet averaged over IOU threshold values 0.50, 0.55, ... 0.95.
        """
        
        if isTraining:
            areaMaxDetInfo = [ (0,      np.inf, 100, 'Area: All      MaxDet: 100') ]
        else:
            areaMaxDetInfo = [ (0,      np.inf, 100, 'Area: All      MaxDet: 100'),
                               (0,      32*32., 100, 'Area: Small    MaxDet: 100'),
                               (32*32., 96*96., 100, 'Area: Medium   MaxDet: 100'),
                               (96*96., np.inf, 100, 'Area: Large    MaxDet: 100'),
                               (0,      np.inf, 1,   'Area: All      MaxDet: 1  '),
                               (0,      np.inf, 10,  'Area: All      MaxDet: 10 ') ]
        iouMins = np.arange(.5,1,.05)

        startTime = time.time()
        if quiet == False: myPrint('Evaluating inference results for %d images ... '%(len(results)))
        ious = self.calculateIous(results, quiet=quiet)
        numGts, dtScores, dtMatches = self.findMatches(results, ious, iouMins, areaMaxDetInfo, quiet=quiet)
        ap50, ap75, ap, ar = self.calStats(numGts, dtScores, dtMatches, quiet=quiet)
        if isTraining:
            if not quiet:
                myPrint('Done (%.1f Seconds)\n'%(time.time()-startTime))
                myPrint('mAP = %.3f (IoU=0.50:0.95, All Area Sizes, MaxDet: 100)\n'%(ap[0]))
            return ap[0]
        
        resultsStrs = [
                        "Average Precision (AP):",
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[0][3], ap[0] ),
                        '    IoU=0.50        %s  = %.3f'%( areaMaxDetInfo[0][3], ap50[0]),
                        '    IoU=0.75        %s  = %.3f'%( areaMaxDetInfo[0][3], ap75[0]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[1][3], ap[1]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[2][3], ap[2]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[3][3], ap[3]),
                        "Average Recall (AR):",
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[4][3], ar[4]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[5][3], ar[5]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[0][3], ar[0]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[1][3], ar[1]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[2][3], ar[2]),
                        '    IoU=0.50:0.95   %s  = %.3f'%( areaMaxDetInfo[3][3], ar[3])
                      ]
                      
        if not quiet:
            myPrint('Done (%.1f Seconds)\n'%(time.time()-startTime))
            for line in resultsStrs: print(line)
        
        return ap, ap50, ap75, ar, resultsStrs

    # ******************************************************************************************************************
    def evaluateModel(self, model, batchSize=None, quiet=False, returnMetric=False, **kwargs):
        r"""
        This function evaluates the specified model using this dataset.
        
        Parameters
        ----------
        model : Fireball Model object
            The model being evaluated.
            
        batchSize : int
            The batchSize used for the evaluation process. This function processes one batch of the samples at a time. If this is None, the batch size specified for this dataset object in the __init__ function is used instead.

        quiet : Boolean
            If true, no messages are printed to the "stdout" during the evaluation process.

        returnMetric : Boolean
            If true, instead of calculating all the results, just calculates the main metric of the dataset and returns that. This is mostly used during the training at the end of each epoch.
            
            Otherwise, if this is False (the default), the full results are calculated and a dictionary of all results is returned.

        **kwargs : dict
            This contains some additional task specific arguments. Here is
            a list of what can be included in this dictionary.
            
                * **maxSamples (int)**: The max number of samples from this dataSet to be processed for the evaluation of the model. If not specified, all samples are used (default behavior).
                    
        Returns
        -------
        If returnMetric is True, the actual value of dataset's main metric (mAP) is returned.
        Otherwise, this function returns a dictionary containing the results of the evaluation process.
        """
        maxSamples=kwargs.get('maxSamples', None)

        t0 = time.time()
        if maxSamples is None:  maxSamples = self.numSamples
        if batchSize is None:   batchSize = self.batchSize
        quietProcess = quiet or returnMetric
        totalSamples = 0

        inferResults = []
        totalTime = 0 # If batchSize is 1, we want to calculate the average inference time per sample.
        for b, (batchSamples, batchLabels) in enumerate(self.batches(batchSize)):
            if totalSamples>=maxSamples: break
            if returnMetric and (not quiet):
                model.updateTrainingTable('  Running Inference for %s sample %d ... '%(self.dsName.lower(), totalSamples))
            if not quietProcess:
                if batchSize==1:
                    myPrint('\r  Processing sample %d ... '%(b+1), False)
                else:
                    myPrint('\r  Processing batch %d - (Total Samples so far: %d) ... '%(b+1, totalSamples), False)
            totalSamples += batchSamples.shape[0]
            if batchSize == 1: t0 = time.time()
            dtClasses, dtBoxes, dtScores, dtNums = model.inferBatch(batchSamples)
            n = len(dtNums)
            for i in range(n):
                imgId, (imgW, imgH) = batchLabels[i]

                numObjects = dtNums[i]
                dtResults = ( np.int32(dtClasses[i][:numObjects]),
                              np.float32(dtBoxes[i][:numObjects]*[imgW,imgH,imgW,imgH]),
                              dtScores[i][:numObjects] )
                inferResults += [ (imgId, dtResults) ]
            if b>0: totalTime +=(time.time()-t0)    # Do not count the first sample
            
        if not quietProcess:
            if batchSize==1:
                myPrint('\r  Processed %d Sample. (Time Per Sample: %.2f ms)%30s\n'%(totalSamples, (1000.0*totalTime)/(maxSamples-1),' '))
            else:
                myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)%30s\n'%(totalSamples, time.time()-t0,' '))

        if returnMetric:
            return 100.0 * self.evaluate(inferResults, isTraining=returnMetric, quiet=quietProcess)
            
        ap, ap50, ap75, ar, resultsStrs = self.evaluate(inferResults, isTraining=returnMetric, quiet=quietProcess)
        
        results = {
                    'AP(.5-.95,All,100)': ap[0],
                    'AP(.5,All,100)': ap50[0],
                    'AP(.75,All,100)': ap75[0],
                    'AP(.5-.95,Small,100)': ap[1],
                    'AP(.5-.95,Medium,100)': ap[2],
                    'AP(.5-.95,Large,100)': ap[3],
                    'AR(.5-.95,All,1)': ar[4],
                    'AR(.5-.95,All,10)': ar[5],
                    'AR(.5-.95,All,100)': ar[0],
                    'AR(.5-.95,Small,100)': ar[1],
                    'AR(.5-.95,Medium,100)': ar[2],
                    'AR(.5-.95,Large,100)': ar[3],
                    
                    'csvItems': [ 'AP(.5-.95,All,100)', 'AP(.5,All,100)', 'AP(.75,All,100)',
                                  'AP(.5-.95,Small,100)', 'AP(.5-.95,Medium,100)', 'AP(.5-.95,Large,100)',
                                  'AR(.5-.95,All,1)', 'AR(.5-.95,All,10)', 'AR(.5-.95,All,100)',
                                  'AR(.5-.95,Small,100)', 'AR(.5-.95,Medium,100)', 'AR(.5-.95,Large,100)' ]
                  }
                  
        if model.bestMetric is not None:
            results['best%s'%(self.evalMetricName)] = model.bestMetric
            results['bestEpoch'] = model.bestEpochInfo[0]+1
            results['trainTime'] = model.trainTime
            results['csvItems'] += ['best%s'%(self.evalMetricName),'bestEpoch','trainTime']

        return results
