# Copyright (c) 2020 InterDigital AI Research Lab
"""
This module contains the implementation of "CifarDSet" dataset class for image Classification (`CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_). Use the ``CifarDSetUnitTest.py`` file in the ``UnitTest/Datasets`` folder to run the Unit Test of this implementation.

This implementation assumes that the following files exist in the 'dataPath' directory:

* meta: Class Names.
* train: Training Samples and Labels.
* test: Test Samples and Labels.

**Dataset Stats**
    The 100 classes in the CIFAR-100 dataset are grouped into 20 superclasses. In the following table and in the API, "Fine" is used for the case where the 100-class version is used and "Coarse" is use where the 20 superclasses are used.

    =========   =============   ===========================
    Dataset     Total Samples   Samples Per Class
    =========   =============   ===========================
    Training    50000           500 (Fine), 2500 (Coarse)
    Test        10000           100 (Fine), 500 (Coarse)
    =========   =============   ===========================
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 03/02/2020    Shahab Hamidi-Rad       Created the file.
# 04/15/2020    Shahab Hamidi-Rad       Changed the way datasets are created. Added support for Fine-Tuning datasets,
#                                       merging, splitting, and creating permanent fine-tune datasets.
# 04/23/2020    Shahab Hamidi-Rad       Completed the documentation.
# **********************************************************************************************************************
import pickle
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint

# **********************************************************************************************************************
class CifarDSet(BaseDSet):
    """
    This class implements the CIFAR-100 dataset.
    """
    classNames = None
    numClasses = None
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=64, coarseLabels=False):
        r"""
        Constructs a CifarDSet instance. This can be called directly or via `makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", or "Tune". Note that "Valid" cannot be used here.

        dataPath : str
            The path to the directory where the dataset files are located.

        samples : numpy array or None
            If specified, it is used as the samples for the dataset. It is a numpy array of samples. Each sample is an image represented as a numpy array of shape (32,32,3).
            
        labels : numpy array or None
            If specified, it is a numpy array of int32 values. Each label is an int32 number between 0 and 99 (0 and 19 for the coarse dataset) indicating the class for each sample.
            
        batchSize : int
            The default batch size used in the "batches" method.
            
        coarseLabels : Boolean
            If True, the coarse dataset is loaded which has only 20 classes of images.
        """
        
        if samples is None: # If samples and labels are given, we don't need dataPath and it can be None.
            if dataPath is None:
                dataPath = '/data/CIFAR-100/'
                if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
            assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
        self.coarseLabels = coarseLabels
        super().__init__(dsName, dataPath, samples, labels, batchSize)
        self.dsName = self.dsName.split('%')[0]     # Remove the percentage info now that we don't need it anymore

    # ******************************************************************************************************************
    @classmethod
    def getSamplesAndLabels(cls, fileName, coarseLabels):
        """
        This class method loads the samples and labels form the dataset files.
        
        Parameters
        ----------
        fileName : str
            The dataset file name containing the sample/label information.
        
        coarseLabels : Boolean
            If True, the coarse dataset is loaded which has only 20 classes of images.

        Returns
        -------
        samples : numpy array
            The dataset samples. Each sample is an image represented as a numpy array
            of shape (32,32,3).
        
        labels : numpy array
            The dataset labels. Each label is an int32 number between 0 and 99 (0 and
            19 for the coarse dataset) indicating the class for each sample.
        """
        
        with open(fileName, 'rb') as pickleFile:
            dataDic = pickle.load(pickleFile, encoding='latin1')
        samples = dataDic['data']
        labels = dataDic['coarse_labels' if coarseLabels else 'fine_labels']
        return samples, np.array(labels)

    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This function is called by the constructor of the base dataset class to load the samples and labels of this dataset based on `dsName` and `dataPath` properties of this class.
        
        Notes
        -----
        The dsName "Valid" cannot be used here. A validation dataset should be created using the `makeDatasets` method or using the `split` method on an existing training dataset.
        """

        if CifarDSet.classNames is None:
            with open(self.dataPath+'meta', 'rb') as pickleFile:
                metaDic = pickle.load(pickleFile, encoding='latin1')
                CifarDSet.classNames = metaDic['coarse_label_names'] if self.coarseLabels else metaDic['fine_label_names']
                CifarDSet.numClasses = len(CifarDSet.classNames)

        if 'valid' in self.dsName.lower():
            raise ValueError("Validation dataset can only be created using 'makeDatasets' or 'split' methods!")

        elif 'train' in self.dsName.lower():
            self.samples, self.labels = self.getSamplesAndLabels(self.dataPath+'train', self.coarseLabels)
            self.samples = np.transpose( self.samples.reshape(-1,3,32,32), (0,2,3,1) )
        
        elif 'tun' in self.dsName.lower():
            if ('%' not in self.dsName) and os.path.exists(self.dataPath + 'TuneDb.npz'):
                dataset = np.load(self.dataPath + 'TuneDb.npz')
                self.samples = dataset['samples']
                self.labels = dataset['labels']
            else:
                if '%r' in self.dsName:     ratio = float(self.dsName.split('%r')[1])/100.0 # Random
                elif '%' in self.dsName:    ratio = float(self.dsName.split('%')[1])/100.0  # Repeatable
                else:                       ratio = 0.1
                
                trainSamples, trainLabels = self.getSamplesAndLabels(self.dataPath+'train', self.coarseLabels)
                trainSamples = np.transpose( trainSamples.reshape(-1,3,32,32), (0,2,3,1) )
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
            self.samples, self.labels = self.getSamplesAndLabels(self.dataPath+'test', self.coarseLabels)
            self.samples = np.transpose( self.samples.reshape(-1,3,32,32), (0,2,3,1) )
            
        else:
            raise ValueError("Unknown dataset name \"%s\"!"%(self.dsName))

    # ******************************************************************************************************************
    def split(self, dsName='Valid', ratio=.1, batchSize=None, repeatable=True):
        r"""
        This function splits the current dataset and returns a portion of data as a new `CifarDSet` object. The current object is then updated to keep the remaining samples.
        
        This method keeps the same ratio between the number of samples for each class. This means if the original dataset was not balanced, the split datasets also are not balanced and they have the same ratio of number of samples per class.
        
        Parameters
        ----------
        dsName : str
            The name of the new dataset that is created.

        ratio : float
            The ratio between the number of samples that are removed from this dataset to the total number of the samples in this dataset before the split. The default value of 0.1 results in creating a new dataset with %10 of the samples. The remaining %90 of the samples stay in the current instance.

        batchSize : int or None
            The batchSize used for the new `CifarDSet` object created. If not specified the new `CifarDSet` instance inherits the batchSize from this object.

        repeatable : Boolean
            If True, the sampling from the original dataset is deterministic and therefore the experiments are repeatable. Otherwise, the sampling is done randomly.

        Returns
        -------
        CifarDSet
            A new dataset containing a portion (specified by `ratio`) of samples from this object.
        """
        
        splitIndexes = self.getSplitIndexes(ratio, repeatable)

        # Make sure the new dataset is of the same class as self. This is necessary for derived
        # classes.
        cls = self.__class__
        splitDSet = cls(dsName, samples=self.samples[splitIndexes], labels=self.labels[splitIndexes],
                        batchSize=batchSize, coarseLabels=self.coarseLabels)

        # Update myself with the remaining samples:
        remainingIndexes = np.setdiff1d(range(self.numSamples), splitIndexes)
        self.samples = self.samples[remainingIndexes]
        self.labels = self.labels[remainingIndexes]
        self.numSamples = len(self.samples)
        self.sampleIndexes = np.arange(self.numSamples)

        return splitDSet

    # ******************************************************************************************************************
    @classmethod
    def createFineTuneDataset(cls, dataPath=None, ratio=0.1, coarseLabels=False):
        r"""
        This class method creates a fine-tuning dataset and saves the information permanently on file system so that it can be loaded later.
        
        Parameters
        ----------
        dataPath : str
            The path to the directory where the dataset files are located.

        ratio : float
            The ratio between the number of samples in the fine-tuning dataset and total training samples. The default value of 0.1 results in creating a new dataset with %10 of the training samples.
        
        coarseLabels : Boolean
            If True, the coarse dataset (with 20 classes) is loaded and used for creation of the Fine-Tuning dataset.

        Note
        ----
        This method can be used when consistent results is required. The same dataset samples are used every time the fine-tuning algorithm uses the dataset created by this method.
        """

        if dataPath is None:
            dataPath = '/data/CIFAR-100/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        if CifarDSet.classNames is None:
            with open(dataPath+'meta', 'rb') as pickleFile:
                metaDic = pickle.load(pickleFile, encoding='latin1')
                CifarDSet.classNames = metaDic['coarse_label_names'] if coarseLabels else metaDic['fine_label_names']
                CifarDSet.numClasses = len(CifarDSet.classNames)

        trainSamples, trainLabels = cls.getSamplesAndLabels(dataPath+'train', coarseLabels)
        trainSamples = np.transpose( trainSamples.reshape(-1,3,32,32), (0,2,3,1) )
        numTrainSamples = len(trainLabels)
        
        numTuneSamples = int(np.round(numTrainSamples*ratio))
        numPerClass = int(np.round(numTuneSamples/cls.numClasses))
        numTuneSamples = numPerClass*cls.numClasses

        myPrint('Creating Fine-Tune Dataset with %d samples ... '%(numTuneSamples), False)
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
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=64, dataPath=None, coarseLabels=False):
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
                           
                If "Tune" is included instead of "Train", then the validation samples are taken from the fine-tuning samples. (See the examples below.)
            
            * **"Tune"**:  Create the Fine-Tuning datas. The ratio of fine-tuning samples can be specified using a % sign followed by the percentage. For example "Tune%5" means create a fine-tuning dataset with %5 of the training datas.
                           
                If a percentage is not specified and a Tuning dataset file (created by `createFineTuneDataset` function) is available, the fine-tuning samples are loaded from the existing file.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        coarseLabels : Boolean
            If True, the coarse datasets are loaded which have only 20 classes of images.

        Returns
        -------
        Up to 3 `CifarDSet` objects
            Depending on the number of items specified in the `dsNames`, it returns between one and three `CifarDSet` objects. The returned values have the same order as they appear in the `dsNames` parameter.

        Note
        ----
        * To specify the training dataset, any string containing the word "train" (case insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        * To specify the test dataset, any string containing the word "test" (case insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        * To specify the validation dataset, any string containing the word "valid" (case insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.
        * To specify the fine-tuning dataset, any string containing the word "tun" (case insensitive) is accepted. So, "Fine-Tuning", "Tuning", and 'tune' all can be used.

        When the '%' is used to specify the ratio for 'Validation' and 'Tuning' datasets, the subsampling is deterministic and the results are repeatable across different executions and even different platforms. If you want the results to be random, you can use '%r' instead of '%'. For example "Tune%r10" creates a dataset with %10 of training data which are selected randomly. A different call on the same or different machine will probably choose a different set of samples.
        
        Examples
        --------
        * ``dsNames="Train,Test,Valid%5"``: 3 `CifarDSet` objects are returned for training, test, and validation in the same order. The validation dataset contains %5 of available training data and training dataset contains the remaining %95.
        * ``dsNames="Train,Test"``: 2 `CifarDSet` objects are returned for training and test. The training dataset contains all available training data.
        * ``dsNames="FineTuning%r5,Test"``: 2 `CifarDSet` objects are returned for fine-tuning and test. The fine-tuning dataset contains %5 of the training data (picked randomly because of '%r')
        * ``dsNames="Tune%5,Test,Validation%5"``: 3 `CifarDSet` objects are returned for fine-tuning, test, and validation in the same order. The fine-tuning and validation together contain %5 of available training data. Validation dataset contains %5 of that (0.0025 of training data or %5 of %5) and Fine-Tuning dataset contains the remaining %95 (0.0475 of training data or %95 of %5)
        """

        validSource = None
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            if ('train' in dsStr.lower()) or ('tun' in dsStr.lower()):
                validSource = cls(dsStr, dataPath, batchSize=batchSize, coarseLabels=coarseLabels)
                retVals += [ validSource ]
            elif 'valid' in dsStr.lower():
                assert validSource is not None, "'%s' must follow a 'Train' or a 'Tune' dataset name!"%(dsStr)
                ratio = .1
                if '%r' in dsStr:   ratio = float(dsStr.split('%r')[1])/100.0
                elif '%' in dsStr:  ratio = float(dsStr.split('%')[1])/100.0
                validDs = validSource.split(dsStr, ratio, batchSize, repeatable=('%r' not in dsStr))
                retVals += [ validDs ]
            else:
                retVals += [ cls(dsStr, dataPath, batchSize=batchSize, coarseLabels=coarseLabels) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals
