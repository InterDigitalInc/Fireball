# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset for handwritten digit Classification.
Use the ``MnistDSetUnitTest.py`` file in the ``UnitTest/Datasets`` directory to run the Unit Test of this implementation.

**Dataset Stats**
    =========   =============   =================
    Dataset     Total Samples   Samples Per Class
    =========   =============   =================
    Training    60000           5421 to 6742
    Test        10000           892 to 1135
    =========   =============   =================
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    ------------------------------------------------
# 03/02/2020    Shahab Hamidi-Rad       Created the file.
# 04/15/2020    Shahab Hamidi-Rad       Changed the way datasets are created. Added support
#                                       for Fine-Tuning datasets, merging, and splitting
#                                       datasets.
# 04/23/2020    Shahab Hamidi-Rad       Completed the documentation.
# **********************************************************************************************************************
import struct
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint

# **********************************************************************************************************************
class MnistDSet(BaseDSet):
    """
    This class implements the MNIST dataset.
    """
    classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    numClasses = len(classNames)
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=64):
        r"""
        Constructs an `MnistDSet` instance. This can be called directly or via `makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", or "Tune". Note that "Valid" cannot be used here.

        dataPath : str
            The path to the directory where the dataset files are located. This implementation expects the following files in the "dataPath" directory:
            
            * t10k-images.idx3-ubyte
            * t10k-labels.idx1-ubyte
            * train-images.idx3-ubyte
            * train-labels.idx1-ubyte

        samples : numpy array or None
            If specified, it is used as the samples for the dataset. It is a numpy array of samples. Each sample is an image represented as a numpy array of shape (28,28,1).
            
        labels : numpy array or None
            If specified, it is a numpy array of int32 values. Each label is an int32 number between 0 and 9 indicating the class for each sample.
            
        batchSize : int
            The default batch size used in the "batches" method.
        """
        
        if samples is None: # If samples and labels are given, we don't need dataPath and it can be None.
            if dataPath is None:
                dataPath = '/data/mnist/'
                if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
            assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        super().__init__(dsName, dataPath, samples, labels, batchSize)
        self.dsName = self.dsName.split('%')[0]     # Remove the percentage info now that we don't need it anymore

    # ******************************************************************************************************************
    @classmethod
    def getMnistImages(cls, fileName):
        r"""
        Loads the samples from the specified MNIST dataset file.
        
        Parameters
        ----------
        fileName : str
            The name of dataset file.
            
        Returns
        -------
        numpy array
            The samples loaded from the specified file. Each sample are numpy arrays of type float32 and shape (28, 28, 1). All the samples are normalized between -1 and 1.
        """
        
        file = open(fileName, mode='rb')
        header = file.read(16)
        magic, numImages, imageWidth, imageHeight = struct.unpack(">iiii", header)
        assert magic == 2051, "Error: Invalid MNIST Image format!"

        buf = file.read(imageWidth * imageHeight * numImages)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data /= 255.0   # Normalize to [0..1]
        data = (data-127.5)/255.0   # Normalize to [-1..1]
        return data.reshape(numImages, imageWidth, imageHeight, 1)

    # ******************************************************************************************************************
    @classmethod
    def getMnistLabels(cls, fileName):
        r"""
        Loads the labels from the specified MNIST dataset file.
        
        Parameters
        ----------
        fileName : str
            The name of dataset file.
            
        Returns
        -------
        numpy array
            The label values.
        """

        file = open(fileName, mode='rb')
        header = file.read(8)
        magic, numLabels = struct.unpack(">ii", header)
        assert magic == 2049, "Error: Invalid MNIST Label format!"

        buf = file.read(numLabels)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

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

        elif 'train' in self.dsName.lower():
            self.samples = MnistDSet.getMnistImages(self.dataPath + 'train-images.idx3-ubyte')
            self.labels = MnistDSet.getMnistLabels(self.dataPath + 'train-labels.idx1-ubyte')
        
        elif 'tun' in self.dsName.lower():
            if ('%' not in self.dsName) and os.path.exists(self.dataPath + 'TuneDb.npz'):
                dataset = np.load(self.dataPath + 'TuneDb.npz')
                self.samples = dataset['samples']
                self.labels = dataset['labels']
            else:
                if '%r' in self.dsName:     ratio = float(self.dsName.split('%r')[1])/100.0 # Random
                elif '%' in self.dsName:    ratio = float(self.dsName.split('%')[1])/100.0  # Repeatable
                else:                       ratio = 0.1
                trainSamples = MnistDSet.getMnistImages(self.dataPath + 'train-images.idx3-ubyte')
                trainLabels = MnistDSet.getMnistLabels(self.dataPath + 'train-labels.idx1-ubyte')
                numTuneSamples = int(np.round(len(trainLabels)*ratio))
                numPerClass = int(np.round(numTuneSamples/self.numClasses))
                numTuneSamples = numPerClass*self.numClasses

                tuneSamples = []
                tuneLabels = []
                for c in range(self.numClasses):
                    cIndexes = np.where(trainLabels==c)[0]
                    if '%r' in self.dsName:
                        sampleIndexes = sorted(np.random.choice(cIndexes, numPerClass, replace=False))  # Random
                    else:
                        sampleIndexes = cIndexes[:numPerClass]                                          # Repeatable
                    tuneSamples += [ trainSamples[ sampleIndexes ] ]
                    tuneLabels += [c]*numPerClass
                self.samples = np.concatenate(tuneSamples)
                self.labels = np.array(tuneLabels)
                
        elif 'test' in self.dsName.lower():
            self.samples = MnistDSet.getMnistImages(self.dataPath + 't10k-images.idx3-ubyte')
            self.labels = MnistDSet.getMnistLabels(self.dataPath + 't10k-labels.idx1-ubyte')
            
        else:
            raise ValueError("Unknown dataset name \"%s\"!"%(self.dsName))

    # ******************************************************************************************************************
    def split(self, dsName='Valid', ratio=.1, batchSize=None, repeatable=True):
        r"""
        This function splits the current dataset and returns a portion of data as a new `MnistDSet` object. The current object is then updated to keep the remaining samples.
        
        This method keeps the same ratio between the number of samples for each class. This means if the original dataset was not balanced, the split datasets also are not balanced and they have the same ratio of number of samples per class.
        
        Parameters
        ----------
        dsName : str
            The name of the new dataset that is created.

        ratio : float
            The ratio between the number of samples that are removed from this dataset to the total number of the samples in this dataset before the split. The default value of 0.1 results in creating a new dataset with %10 of the samples. The remaining %90 of the samples stay in the current instance.

        batchSize : int or None
            The batchSize used for the new `MnistDSet` object created. If not specified the new `MnistDSet` instance inherits the batchSize from this object.

        repeatable : Boolean
            If True, the sampling from the original dataset is deterministic and therefore the experiments are repeatable. Otherwise, the sampling is done randomly.

        Returns
        -------
        MnistDSet
            A new dataset containing a portion (specified by `ratio`) of samples from this object.
        """
        splitIndexes = self.getSplitIndexes(ratio, repeatable)

        # Make sure the new dataset is of the same class as self. This is necessary for derived
        # classes.
        cls = self.__class__
        splitDSet = cls(dsName, samples=self.samples[splitIndexes], labels=self.labels[splitIndexes],
                        batchSize=batchSize)

        # Update myself with the remaining samples:
        remainingIndexes = np.setdiff1d(range(self.numSamples), splitIndexes)
        self.samples = self.samples[remainingIndexes]
        self.labels = self.labels[remainingIndexes]
        self.numSamples = len(self.samples)
        self.sampleIndexes = np.arange(self.numSamples)

        return splitDSet
    
    # ******************************************************************************************************************
    @classmethod
    def createFineTuneDataset(cls, dataPath=None, ratio=0.1):
        r"""
        This class method creates a fine-tuning dataset and saves the information permanently on file system so that it can be loaded later.
        
        Parameters
        ----------
        dataPath : str
            The path to the directory where the dataset files are located.

        ratio : float
            The ratio between the number of samples in the fine-tuning dataset and total training samples. The default value of 0.1 results in creating a new dataset with %10 of the training samples.
        
        Note
        ----
        This method can be used when consistent results is required. The same dataset samples are used every time the fine-tuning algorithm uses the dataset created by this method.
        """

        if dataPath is None:
            dataPath = '/data/mnist/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
        
        trainSamples = MnistDSet.getMnistImages(dataPath + 'train-images.idx3-ubyte')
        trainLabels = MnistDSet.getMnistLabels(dataPath + 'train-labels.idx1-ubyte')
        numTrainSamples = len(trainLabels)
        
        numTuneSamples = int(np.round(numTrainSamples*ratio))
        numPerClass = int(np.round(numTuneSamples/cls.numClasses))
        numTuneSamples = numPerClass*cls.numClasses

        myPrint('Creating Fine Tune Dataset with %d samples ... '%(numTuneSamples), False)
        t0 = time.time()
        tuneSamples = []
        tuneLabels = []

        for c in range(cls.numClasses):
            cIndexes = np.where(trainLabels==c)[0]
            sampleIndexes = sorted(np.random.choice(cIndexes, numPerClass, replace=False))

            tuneSamples += [ trainSamples[ sampleIndexes ] ]
            tuneLabels += [c]*numPerClass
        tuneSamples = np.concatenate(tuneSamples)
        tuneLabels = np.array(tuneLabels)

        rootDic = { 'samples': tuneSamples, 'labels': tuneLabels }
        np.savez_compressed(dataPath+'TuneDb.npz', **rootDic)
        myPrint('Done. (%.2f Seconds)'%(time.time()-t0))

    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=32, dataPath=None):
        """
        This class method creates several datasets as specified by `dsNames` parameter in one-shot.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
            
            * **"Train"**: Create the training dataset.
            * **"Test"**:  Create the test dataset.
            * **"Valid"**: Create the validation dataset. The ratio of validation samples can be specified using a % sign followed by the percentage. For example "Valid%10" means create a validation dataset with %10 of the training datas.
                           
                If "Valid" is included in `dsNames`, there must be at least a "Train" or "Tune" dataset.
                           
                If "Tune" is included instead of "Train", then the validation samples are taken from the fine-tuning samples. (See the examples below.)
            * **"Tune"**:  Create the Fine-Tuning datas. The ratio of fine-tuning samples can be specified using a % sign followed by the percentage. For example "Tune%5" means create a fine-tuning dataset with %5 of the training datas.
                
                If a percentage is not specified and a Tuning dataset file (created by `createFineTuneDataset` function) is available, the fine-tuning samples are loaded from the existing file.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        Returns
        -------
        Up to 3 `MnistDSet` objects
            Depending on the number of items specified in the `dsNames`, it returns between one and three `MnistDSet` objects. The returned values have the same order as they appear in the `dsNames` parameter.

        Note
        ----
        * To specify the training dataset, any string containing the word "train" (case insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        * To specify the test dataset, any string containing the word "test" (case insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        * To specify the validation dataset, any string containing the word "valid" (case insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.
        * To specify the fine-tuning dataset, any string containing the word "tun" (case insensitive) is accepted. So, "Fine-Tuning", "Tuning", and 'tune' all can be used.

        When the '%' is used to specify the ratio for 'Validation' and 'Tuning' datasets, the subsampling is deterministic and the results are repeatable across different executions and even different platforms. If you want the results to be random, you can use '%r' instead of '%'. For example "Tune%r10" creates a dataset with %10 of training data which are selected randomly. A different call on the same or different machine will probably choose a different set of samples.
        
        Examples
        --------
        * ``dsNames="Train,Test,Valid%5"``: 3 `MnistDSet` objects are returned for training, test, and validation in the same order. The validation dataset contains %5 of available training data and training dataset contains the remaining %95.
        * ``dsNames="Train,Test"``: 2 `MnistDSet` objects are returned for training and test. The training dataset contains all available training data.
        * ``dsNames="FineTuning%r5,Test"``: 2 `MnistDSet` objects are returned for fine-tuning and test. The fine-tuning dataset contains %5 of the training data (picked randomly because of '%r')
        * ``dsNames="Tune%5,Test,Validation%5"``: 3 `MnistDSet` objects are returned for fine-tuning, test, and validation in the same order. The fine-tuning and validation together contain %5 of available training data. Validation dataset contains %5 of that (0.0025 of training data or %5 of %5) and Fine-Tuning dataset contains the remaining %95 (0.0475 of training data or %95 of %5)
        """

        validSource = None
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            if ('train' in dsStr.lower()) or ('tun' in dsStr.lower()):
                validSource = cls(dsStr, dataPath, batchSize=batchSize)
                retVals += [ validSource ]
            elif 'valid' in dsStr.lower():
                assert validSource is not None, "'%s' must follow a 'Train' or a 'Tune' dataset name!"%(dsStr)
                ratio = .1
                if '%r' in dsStr:   ratio = float(dsStr.split('%r')[1])/100.0
                elif '%' in dsStr:  ratio = float(dsStr.split('%')[1])/100.0
                validDs = validSource.split(dsStr, ratio, batchSize)
                retVals += [ validDs ]
            else:
                retVals += [ cls(dsStr, dataPath, batchSize=batchSize) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals
