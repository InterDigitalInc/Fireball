# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of `ImageNet <http://image-net.org/index>`_ dataset for Image Recognition. This implementation assumes that the following files exist in the 'dataPath' directory:
    
    * ``ILSVRC2012Train224``: Contains 1000 folders (one per class named by the class id). The training image files in each folder are named like "Image_nnnnnn.JPEG" with nnnnnn starting at 000001.
    * ``ILSVRC2012Val224``: Contains 1000 folders (one per class named by the class id). The validation image files in each folder are named like "Image_nnnnnn.JPEG" with nnnnnn starting at 000001.
    * ``TrainDataset.csv``: Information about each class and number of training samples for each class.
    * ``ValDataset.csv``: Information about each class and number of validation samples for each class.

**Pre-Processing**
    * ``Crop256Cafe``: Resize smaller dim to 256, Crop center 224, BGR output, normalized with mean: [103.939, 116.779, 123.68]
    * ``ForceCafe``: Force resize to 224x224, May Lose Aspect Ratio, BGR output, normalized with mean: [103.939, 116.779, 123.68]
    * ``Crop256PyTorch``: Resize smaller dim to 256, Crop center 224, RGB output, normalized to 0..1, then use mean:[0.485, 0.456, 0.406] and var: [0.229, 0.224, 0.225]
    * ``ForcePyTorch``: Force resize to 224x224, May Lose Aspect Ratio, RGB output, normalized to 0..1, then use mean:[0.485, 0.456, 0.406] and var: [0.229, 0.224, 0.225]
    * ``Crop256Tf``: Resize smaller dim to 256, Crop center 224, RGB output, normalized to -1..1
    * ``ForceTf``: Force resize to 224x224, May Lose Aspect Ratio, RGB output, normalized to -1..1

To test this implementation, run the ``ImageNetDSetUnitTest.py`` file in the ``UnitTest/Datasets`` directory.

For more info see `Keras image utilities <https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py>`_ and `preprocessing <https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/_impl/keras/preprocessing/image.py>`_ code.

**Dataset Stats**
    =========   =============   ===========================
    Dataset     Total Samples   Samples Per Class
    =========   =============   ===========================
    Training    1,281,167       732 to 1,300
    Test        50,000          50
    =========   =============   ===========================
"""

# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 03/03/2020              Shahab Hamidi-Rad       Created the file.
# 04/15/2020              Shahab Hamidi-Rad       Changed the constructor signature to match the other
#                                                 datasets. Added support for Fine-Tuning datasets,
#                                                 merging, and splitting datasets.
# 04/17/2020              Shahab Hamidi-Rad       Completed the documentation.
# 12/04/2020              Shahab Hamidi-Rad       The labels[s] are now keeps the class for sample no. 's' instead of
#                                                 samples[s][2].
# 10/11/2021              Shahab Hamidi-Rad       Added support for downloading datasets.
# **********************************************************************************************************************
import numpy as np
import cv2
import os
from .base import BaseDSet
from ..printutils import myPrint
import time

# **********************************************************************************************************************
class ImageNetDSet(BaseDSet):
    r"""
    This class implements the ImageNet dataset.
    """
    
    numClasses = 1000
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=256,
                 preProcessing='Crop256Cafe', numWorkers=8):
        r"""
        Constructs an ImageNetDSet instance. This can be called directly or via `makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", or "Tune". Note that "Valid" cannot be used here.

        dataPath : str
            The path to the directory where the dataset files are located.

        samples : list or None
            If specified it is used as the samples for the dataset. It is a list of tuples containing information about the samples in the dataset. If samples is not specified, the `loadSamples` method is called by the base class.

        labels : None
            The labels for each sample in the dataset.

        batchSize : int
            The default batch size used in the "batches" method.

        preProcessing : str
            The type of preprocessing used when loading the images. Please refer to the "Pre-Processing" section above in this module's documentation for an explanation for each one of the pre-processing methods.

        numWorkers : int
            The number of worker threads used to load the images.
        """
        
        if samples is None: # If samples and labels are given, we don't need dataPath and it can be None.
            if dataPath is None:
                dataPath = '/data/ImageNet/'
                if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
            assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        super().__init__(dsName, dataPath, samples, labels, batchSize, numWorkers)
        self.sampleShape = (224, 224, 3)
        if 'test' in dsName.lower():
            self.imagesFolder = dataPath + 'ILSVRC2012Test224/'
        elif 'tun' in dsName.lower():
            if '%' in dsName:                                       self.imagesFolder = dataPath + 'ILSVRC2012Train224/'
            elif os.path.exists(dataPath + 'ILSVRC2012Tune224/'):   self.imagesFolder = dataPath + 'ILSVRC2012Tune224/'
            else:                                                   self.imagesFolder = dataPath + 'ILSVRC2012Train224/'
        else:
            self.imagesFolder = dataPath + 'ILSVRC2012Train224/'
        
        self.dsName = self.dsName.split('%')[0]     # Remove the percentage info now that we don't need it anymore
        self.preProcessing = preProcessing

    # ******************************************************************************************************************
    @classmethod
    def download(cls, dataFolder=None):
        r"""
        This class method can be called to download the ImageNet dataset files from a Fireball
        online repository. Please note that this does not include the training dataset. Only
        Tuning and Test datasets are downloaded. All image files have already been resized to
        224x224.
        
        Parameters
        ----------
        dataFolder: str
            The folder where dataset files are saved. If this is not provided, then
            a folder named "data" is created in the home directory of the current user and the
            dataset folders and files are created there. In other words, the default data folder
            is ``~/data``
        """
        files = ['ILSVRC2012Tune224.zip',
                 'ILSVRC2012Test224.zip',
                 'TrainDataset.csv',
                 'TuneDataset.csv',
                 'ValDataset.csv']
        BaseDSet.download("ImageNet", files, dataFolder)

    # ******************************************************************************************************************
    def __repr__(self):
        r"""
        Provides a text string containing the information about this object. Calls the same method in the base class to print common information.
        
        Returns
        -------
        str
            The text string containing the information about this object.
        """
        
        repStr = super().__repr__()
        repStr += '    Preprocessing .................................. %s\n'%(self.preProcessing)
        repStr += '    Number of Workers .............................. %s\n'%(str(self.numWorkers))
        return repStr

    # ******************************************************************************************************************
    def getImage(self, fileName):
        r"""
        Returns a numpy array containing the image information for the image file specified by the `fileName`.
        
        Parameters
        ----------
        fileName : str
            The name of the image file.

        Returns
        -------
        numpy array
            The image information as a numpy array. If the `preProcessing` is one of 'Crop256Cafe' or 'ForceCafe', the returned value is in BGR format; otherwise it is in RGB format.
        """
        
        img = cv2.imread(fileName)  # Reads image in BGR order
        if self.preProcessing in ['Crop256PyTorch','ForcePyTorch', 'Crop256Tf', 'ForceTf']:
            img = np.float32(img)[..., ::-1]    # Convert to RGB
        return np.float32(img)

    # ******************************************************************************************************************
    def preProcessImages(self, images):
        r"""
        Preprocesses the specified `image` based on the method specified by `preProcessing`. Please refer to the "Pre-Processing" section above in this module's documentation for an explanation for each one of the pre-processing methods.
        
        Parameters
        ----------
        images : numpy array
            The image(s) to be pre-processed as a numpy array.

        Returns
        -------
        numpy array
            The processed image(s) as a numpy array.
        """
        
        if self.preProcessing in ['Crop256Cafe','ForceCafe']:
            # Image(s) MUST BE float32 Numpy Array(s) in BGR Format
            return (np.float32(images) - [103.939, 116.779, 123.68])

        if self.preProcessing in ['Crop256PyTorch','ForcePyTorch']:
            # Image(s) MUST BE float32 Numpy Array(s) in RGB Format
            return ((np.float32(images)/256.0)-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]

        if self.preProcessing in ['Crop256Tf','ForceTf']:
            # Image(s) MUST BE float32 Numpy Array(s) in RGB Format
            return ((np.float32(images)/127.5)-1.0)
        
        raise ValueError('Unknown preProcessing "%s"!'%(self.preProcessing))

    # ******************************************************************************************************************
    def resizedImg(self, img):
        r"""
        Resizes the specified image using the method specified by `preProcessing`. The resized image is always a 224x224x3 numpy array of type float32.
        
        Parameters
        ----------
        img : numpy array
            The image to be resized.

        Returns
        -------
        numpy array
            The resized image as a numpy array of shape (224,224,3)
        """
        
        if self.preProcessing in ['ForceCafe','ForcePyTorch', 'ForceTf']:
            # Note that this will probably change the aspect ratio of the image
            return cv2.resize(img, (224,224), interpolation = cv2.INTER_NEAREST)

        if self.preProcessing in ['Crop256Cafe','Crop256PyTorch', 'Crop256Tf']:
            imgSize = img.shape[:2]
            ratio = 256.0/min(imgSize)
            newSize = (int(np.round(imgSize[1]*ratio)), int(np.round(imgSize[0]*ratio)))

            # Note: INTER_AREA is best when shrinking and CV_INTER_CUBIC is best when enlarging
            img = cv2.resize(img, newSize,  interpolation = (cv2.INTER_AREA if ratio<1.0 else cv2.INTER_CUBIC))

            # Now crop the center 224x224 image
            dw = newSize[0] - 224
            dh = newSize[1] - 224
            resizedImg = img[dh//2:dh//2+224, dw//2:dw//2+224,:]
            return resizedImg

        raise ValueError('Unknown preProcessing "%s"!'%(self.preProcessing))

    # ******************************************************************************************************************
    def getPreprocessedImage(self, imageFileName):
        r"""
        A utility function that loads an image, resizes and preprocesses it, and returns it as a numpy array of type float32.
        
        Parameters
        ----------
        imageFileName : str
            The path to the image file.

        Returns
        -------
        numpy array
            The resized and preprocessed image as a numpy array of shape (224,224,3)
        """
        
        return self.preProcessImages( self.resizedImg( self.getImage(imageFileName)) )

    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This function is called by the constructor of the base dataset class to load the samples and labels of this dataset based on `dsName` and `dataPath` properties of this class.
        
        Note
        ----
        The dsName "Valid" cannot be used here. A validation dataset should be created using the `makeDatasets` method or using the `split` method on an existing training dataset.
        """
        
        if 'valid' in self.dsName.lower():
            raise ValueError("Validation dataset can only be created using 'makeDatasets' or 'split' methods!")

        classIds = None
        self.samples = []
        self.labels = []

        # Note that the file 'ValDataset.csv' is sorted by its first column (class Index-0 to 999)
        if 'test' in self.dsName.lower():       dsFileName = self.dataPath+'ValDataset.csv'
        elif 'train' in self.dsName.lower():    dsFileName = self.dataPath+'TrainDataset.csv'
        elif 'tun' in self.dsName.lower():
            if '%' in self.dsName:                                  dsFileName = self.dataPath+'TrainDataset.csv'
            elif os.path.exists(self.dataPath+'TuneDataset.csv'):   dsFileName = self.dataPath+'TuneDataset.csv'
            else:                                                   dsFileName = self.dataPath+'TrainDataset.csv'
        else:
            raise ValueError("Unknown dataset name \"%s\"!"%(self.dsName))

        imageNetDs = np.genfromtxt(dsFileName, delimiter=',', dtype=np.string_, skip_header=1)
        imageCounts = np.int32(imageNetDs[:,3])
        classIds = np.vectorize(lambda x:x.decode("utf-8"))(imageNetDs[:,1])
        if ImageNetDSet.classNames is None:
            ImageNetDSet.classNames = (np.vectorize(lambda x:x.decode("utf-8"))(imageNetDs[:,2])).tolist()
        for c in range(self.numClasses):
            self.samples += [(classIds[c], i+1) for i in range(imageCounts[c])]
            self.labels += imageCounts[c]*[c]

        ratio = 0
        if 'tun' in self.dsName.lower():
            if '%' in self.dsName:
                ratio = float(self.dsName.split('%')[1])/100.0
            elif os.path.exists(self.dataPath+'TuneDataset.csv')==False:
                ratio = .05  # Use the default ratio
        
        if ratio>0:
            # For Fine-Tune dataset we always create a balanced dataset
            numTrainSamples = sum(imageCounts)
            numTuneSamples = int(np.round(numTrainSamples * ratio))
            numPerClass = int(np.round(numTuneSamples / self.numClasses))
            numTuneSamples = numPerClass * self.numClasses
        
            self.samples = []
            self.labels = []
            for c in range(self.numClasses):
                nImagesForClass = min(imageCounts[c], numPerClass)
                self.samples += [(classIds[c], i+1) for i in range(nImagesForClass) ]
                self.labels += nImagesForClass*[c]

        self.labels = np.int32(self.labels)

    # ******************************************************************************************************************
    def split(self, dsName='Valid', ratio=.1, batchSize=None):
        r"""
        This function splits the current dataset and returns a portion of data as a new `ImageNetDSet` object. This object is then updated to keep the remaining samples.
        
        This method keeps the same ratio between the number of samples for each class. This means that if the original dataset was not balanced, the split datasets also are not balanced and they have the same ratio of number of samples per class.
        
        
        Parameters
        ----------
        dsName : str
            The name of the new dataset that is created.

        ratio : float
            The ratio between the number of samples that are removed from this dataset to the total number of the samples in this dataset before the split. The default value of .1 results in creating a new dataset with %10 of the samples. %90 of the samples stay in this object.

        batchSize : int or None
            The batchSize used for the new `ImageNetDSet` object created. If not specified the new object inherits the batchSize from this object.

        Returns
        -------
        ImageNetDSet
            A new dataset containing a portion (specified by `ratio`) of samples from this object.

        Note
        ----
        * This function assumes that "self.samples" is organized as follows::
        
            All samples in class 0
            All samples in class 1
                    ...
            All samples in class 999
            
          This is the case if `loadSamples` method is used to load the samples.
          
        * The sampling from the original dataset is deterministic and therefore the experiments are repeatable.
        """

        numSplitSamples = int(np.round(self.numSamples*ratio))
        classStartIndex, curClassIdx = 0, 0
        splitSamples, remainingSamples = [], []
        splitLabels, remainingLabels = [], []
        nSplit, nRemaining = 0, 0
        for s, (classId, imageId) in enumerate(self.samples):
            if (self.labels[s] == curClassIdx) and (s<(self.numSamples-1)): continue
            nSampleForClass = s - classStartIndex
            if s == (self.numSamples-1):
                nSampleForClass += 1
                nSampleForClassSplit = numSplitSamples - nSplit
            else:
                dynamicRatio = float(numSplitSamples-nSplit)/(self.numSamples-nSplit-nRemaining)
                nSampleForClassSplit = int(np.round(nSampleForClass * dynamicRatio))

            nSplit += nSampleForClassSplit
            nRemaining += nSampleForClass - nSampleForClassSplit

            splitSamples += self.samples[classStartIndex : classStartIndex+nSampleForClassSplit]
            splitLabels += self.labels[classStartIndex : classStartIndex+nSampleForClassSplit].tolist()
            remainingSamples += self.samples[classStartIndex+nSampleForClassSplit : classStartIndex+nSampleForClass]
            remainingLabels += self.labels[classStartIndex+nSampleForClassSplit : classStartIndex+nSampleForClass].tolist()
            
            classStartIndex = s
            curClassIdx = self.labels[s]
            
        if batchSize is None:   batchSize = self.batchSize
        splitDSet = ImageNetDSet(dsName, self.dataPath, samples=splitSamples, labels=np.int32(splitLabels),
                                 batchSize=batchSize, preProcessing=self.preProcessing, numWorkers=self.numWorkers)

        # Update myself with the remaining samples:
        self.samples = remainingSamples
        self.labels = np.int32(remainingLabels)
        self.numSamples = len(self.samples)
        self.sampleIndexes = np.arange(self.numSamples)

        return splitDSet

    # ******************************************************************************************************************
    @classmethod
    def createFineTuneDataset(cls, dataPath=None, ratio=0.1, copyImages=True):
        r"""
        This class method creates a fine-tuning dataset and saves the information permanently on file system so that it can be loaded later.
        
        Parameters
        ----------
        dataPath : str
            The path to the directory where the dataset files are located.

        ratio : float
            The ratio between the number of samples in the fine-tuning dataset and total training samples.
            The default value of .1 results in creating a new dataset with %10 of the training samples.

        copyImages : Boolean
            If true, copies the images from the training images directory to the new directory "ILSVRC2012Tune224". This is useful when access to the whole dataset is not required for fine-tuning after compression. If this is false, then the images are not copied. The new dataset reuses the original images in the training dataset.
        
        Note
        ----
        This method can be used when consistent results is required. The same dataset samples are used every time the fine-tuning algorithm uses the dataset created by this method.
        """
        
        if dataPath is None:
            dataPath = '/data/ImageNet/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        if copyImages:
            assert os.path.exists(dataPath+'ILSVRC2012Tune224')==False, "A FineTune dataset already exists!"
            os.makedirs(dataPath+'ILSVRC2012Tune224')

        dsFileName = dataPath+'TrainDataset.csv'
        imageNetTrainDs = np.genfromtxt(dsFileName, delimiter=',', dtype=np.string_, skip_header=1)
        imageCountsTrain = np.int32(imageNetTrainDs[:,3])

        classIds = np.vectorize(lambda x:x.decode("utf-8"))(imageNetTrainDs[:,1])
        if cls.classNames is None:  cls.classNames = (np.vectorize(lambda x:x.decode("utf-8"))(imageNetTrainDs[:,2])).tolist()
        numTrainSamples = sum(imageCountsTrain)

        numTuneSamples = int(np.round(numTrainSamples*ratio))
        numPerClass = int(np.round(numTuneSamples/cls.numClasses))
        numTuneSamples = numPerClass*cls.numClasses
        
        myPrint('Creating Fine Tune Dataset with %d samples ... '%(numTuneSamples), copyImages)
        t0 = time.time()
        tuneSamples = []
        tuneDatasetFile = open(dataPath+'TuneDataset.csv', 'w')
        tuneDatasetFile.write('Label,Id,LabelName,numFiles\n')
        nTune, nTrain = 0,0
        for c in range(cls.numClasses):
            if copyImages:
                sampleIndexes = sorted(np.random.choice(imageCountsTrain[c], numPerClass, replace=False))
                os.mkdir(dataPath+'ILSVRC2012Tune224/'+classIds[c])
                for j,i in enumerate(sampleIndexes):
                    percentDone = 100 if (c==(cls.numClasses-1) and j==(numPerClass-1)) else c*100//cls.numClasses
                    myPrint('    Copying "%sILSVRC2012Train224/%s/Image_%06d.JPEG" %d%%\r'%(dataPath,classIds[c],i,
                                                                                            percentDone), False)
                    os.system('cp ' +
                              '%sILSVRC2012Train224/%s/Image_%06d.JPEG '%(dataPath,classIds[c],i+1) +
                              '%sILSVRC2012Tune224/%s/Image_%06d.JPEG'%(dataPath,classIds[c],j+1))

            tuneDatasetFile.write('%d,%s,%s,%d\n'%(c, classIds[c], cls.classNames[c], numPerClass))
        tuneDatasetFile.close()
        myPrint('%sDone. (%.2f Seconds)'%('\n' if copyImages else '', int(time.time()-t0)))

    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=256, dataPath=None, preProcessing='Crop256Cafe', numWorkers=8):
        r"""
        This class method creates several datasets as specified by `dsNames` parameter in one-shot.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
            
            * **"Train"**: Create the training dataset.
            * **"Test"**:  Create the test dataset.
            * **"Valid"**: Create the validation dataset. The ratio of validation samples can be specified using a % sign followed by the percentage. For example "Valid%10" means create a validation dataset with %10 of the training datas.
                         
                If "Valid" is included in `dsNames`, it must be after the "Train" or "Tune".
                
                If "Tune" is included instead of "Train", then the validation samples are taken from the fine-tuning samples. See the examples below.
            * **"Tune"**: Create the Fine-Tuning datas. The ratio of fine-tuning samples can be specified using a % sign followed by the percentage. For example "Tune%5" means create a fine-tuning dataset with %5 of the training datas.
            
                If a percentage is not specified and a Tuning dataset (created by `createFineTuneDataset` function) is available in the `dataPath` directory, the fine-tuning samples are loaded from the existing Tuning dataset.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        preProcessing : str
            The type of preprocessing used when loading the images. Please refer to the "Pre-Processing" section above in this module's documentation for an explanation for each one of the pre-processing methods.
            
        numWorkers : int
            The number of worker threads used to load the images.
            
        Returns
        -------
        Up to 3 ImageNetDSet objects
            Depending on the number of items specified in the `dsNames`, it returns between one and three `ImageNetDSet` objects. The returned values have the same order as they appear in the `dsNames` parameter.

        Notes
        -----
        * To specify the training dataset, any string containing the word "train" (case insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        * To specify the test dataset, any string containing the word "test" (case insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        * To specify the validation dataset, any string containing the word "valid" (case insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.
        * To specify the fine-tuning dataset, any string containing the word "tun" (case insensitive) is accepted. So, "Fine-Tuning", "Tuning", and 'tune' all can be used.

        Examples
        --------
        * ``dsNames="Train,Test,Valid%5"``: 3 `ImageNetDSet` objects are returned for training, test, and validation in the same order. The validation dataset contains %5 of training data and training dataset contains the remaining %95.
        * ``dsNames="Train,Test"``: 2 `ImageNetDSet` objects are returned for training and test.
        * ``dsNames="FineTuning%5,Test"``: 2 `ImageNetDSet` objects are returned for fine-tuning and test. The fine-tuning dataset contains %5 of the training data.
        * ``dsNames="Tune%5,Test,Validation%5"``: 3 `ImageNetDSet` objects are returned for fine-tuning, test, and validation in the same order. The fine-tuning and validation together contain %5 of training data. Validation dataset contains %5 of that (0.0025 of training data or %5 of %5) and Fine-Tuning dataset contains the remaining %95 (0.0475 of training data or %95 of %5)
        """
        
        validSource = None
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            if ('train' in dsStr.lower()) or ('tun' in dsStr.lower()):
                validSource = cls(dsStr, dataPath, batchSize=batchSize, preProcessing=preProcessing, numWorkers=numWorkers)
                retVals += [ validSource ]
            elif 'valid' in dsStr.lower():
                assert validSource is not None, "'%s' must follow a 'Train' or a 'Tune' dataset name!"%(dsStr)
                ratio = .1
                if '%' in dsStr.lower():    ratio = float(dsStr.split('%')[1])/100.0
                validDs = validSource.split(dsStr, ratio, batchSize)
                retVals += [ validDs ]
            else:
                retVals += [ cls(dsStr, dataPath, batchSize=batchSize, preProcessing=preProcessing, numWorkers=numWorkers) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals

    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        r"""
        This method returns a batch of samples and labels from the dataset as specified by the list of indexes in the `batchIndexes` parameter.
        
        Parameters
        ----------
        batchIndexes : list of int
            A list of indexes used to pick samples and labels from the dataset.
            
        Returns
        -------
        samples : numpy array
            The batch samples specified by the `batchIndexes`. Each sample is a resize, pre-processed image as a numpy array of shape (224, 224, 3)
        labels : numpy array
            The batch labels specified by the `batchIndexes`. A numpy array of integer values.
        """
        images = []
        for i in batchIndexes:
            classFolder, imageNo = self.samples[ i ]
            imageFileName = self.imagesFolder + classFolder + ('/Image_%06d.JPEG'%(imageNo))
            images += [ self.getImage( imageFileName ) ]

        return self.preProcessImages(images), self.labels[batchIndexes]
