# Copyright (c) 2020 InterDigital AI Research Lab
"""
************************************************************************************************************************
This file contains the implementation of "AudioDSet" dataset class for Audio Classification.
Use the "AudioDSetUnitTest.py" file to run the Unit Test of this implementation.

This implementation assumes that the following files exist in the 'dataPath' directory:
    TrainDb.npz:        Training data.
    ValidationDb.npz:   Validation data.
    EvalDb.npz:         Evaluation data (used for test dataset if "useEval" is True otherwise ignored)
    TestDb.npz:         Test data (used if "useEval" is True)
    TuneDb.npz:         This can be created using the "createFineTuneDataset" class method.

Dataset Stats:
    training samples ...... 3276, 216-219 samples per class
    test samples .......... 810, 54 samples per class
    eval samples .......... 810, 54 samples per class
    validation samples .... 1404, 93-96 samples per class
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 03/02/2020    Shahab Hamidi-Rad       Created the file.
# 04/15/2020    Shahab Hamidi-Rad       Changed the way datasets are created. Added support for Fine-Tuning datasets,
#                                       merging, splitting, and creating permanent fine-tune datasets.
# 05/26/2020    Shahab Hamidi-Rad       Completed the documentation.
# **********************************************************************************************************************
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint

# **********************************************************************************************************************
class AudioDSet(BaseDSet):
    """
    This class implements the Audio Classification dataset.
    """
    classNames = ['beach' , 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home',
                  'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram' ]
    numClasses = len(classNames)
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=64):
        """
        Constructs a `AudioDSet` instance. This can be called directly or via `makeDatasets`
        class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", "Tune", "Valid", or "Eval".
            
        dataPath : str
            The path to the directory where the dataset files are located.

        samples : numpy array or None
            If specified, it is used as the samples for the dataset. It is a numpy array
            of samples. Each sample is a tensor represented as a numpy array of shape (40, 500, 1).
            
        labels : numpy array or None
            If specified, it is a numpy array of int32 values. Each label is an int32
            number between 0 and 14 indicating the class for each sample.
            
        batchSize : int
            The default batch size used in the "batches" method.
        """
        if samples is None: # If samples and labels are given, we don't need dataPath and it can be None.
            if dataPath is None:
                dataPath = '/data/AudioDb2/'
                if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
            assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        super().__init__(dsName, dataPath, samples, labels, batchSize)
        self.dsName = self.dsName.split('%')[0]     # Remove the percentage info now that we don't need it anymore

    # ******************************************************************************************************************
    def loadSamples(self):
        """
        This function is called by the constructor of the base dataset class to
        load the samples and labels of this dataset based on `dsName` and `dataPath`
        properties of this class.
        """
        if 'train' in self.dsName.lower():
            dataset = np.load(self.dataPath + 'TrainDb.npz')
            self.samples = dataset['samples']
            self.labels = dataset['labels']

        elif 'tun' in self.dsName.lower():
            if ('%' not in self.dsName) and os.path.exists(self.dataPath + 'TuneDb.npz'):
                dataset = np.load(self.dataPath + 'TuneDb.npz')
                self.samples = dataset['samples']
                self.labels = dataset['labels']
            else:
                if '%r' in self.dsName:     ratio = float(self.dsName.split('%r')[1])/100.0 # Random
                elif '%' in self.dsName:    ratio = float(self.dsName.split('%')[1])/100.0  # Repeatable
                else:                       ratio = 0.1
                dataset = np.load(self.dataPath + 'TrainDb.npz')
                trainSamples = dataset['samples']
                trainLabels = dataset['labels']
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
            dataset = np.load(self.dataPath + 'TestDb.npz')
            self.samples = dataset['samples']
            self.labels = dataset['labels']

        elif 'eval' in self.dsName.lower():
            dataset = np.load(self.dataPath + 'EvalDb.npz')
            self.samples = dataset['samples']
            self.labels = dataset['labels']

        elif 'valid' in self.dsName.lower():
            dataset = np.load(self.dataPath + 'ValidationDb.npz')
            self.samples = dataset['samples']
            self.labels = dataset['labels']
        
        else:
            raise ValueError("Unknown dataset name \"%s\"!"%(self.dsName))

    # ******************************************************************************************************************
    def split(self, dsName='Valid', ratio=.1, batchSize=None, repeatable=True):
        """
        This function splits the current dataset and returns a portion of data
        as a new `AudioDSet` object. The current object is then updated to keep the
        remaining samples.
        
        This method keeps the same ratio between the number of samples for each class.
        This means if the original dataset was not balanced, the split datasets
        also are not balanced and they have the same ratio of number of samples per
        class.
        
        Parameters
        ----------
        dsName : str
            The name of the new dataset that is created.

        ratio : float
            The ratio between the number of samples that are removed from this dataset
            to the total number of the samples in this dataset before the split.
            The default value of 0.1 results in creating a new dataset with %10 of the
            samples. The remaining %90 of the samples stay in the current instance.

        batchSize : int or None
            The batchSize used for the new `AudioDSet` object created. If not
            specified the new `AudioDSet` instance inherits the batchSize from
            this object.

        repeatable : bool
            If True, the sampling from the original dataset is deterministic and therefore
            the experiments are repeatable. Otherwise, the sampling is done randomly.

        Returns
        -------
        AudioDSet
            A new dataset containing a portion (specified by `ratio`) of samples
            from this object.
        """
        splitIndexes = self.getSplitIndexes(ratio, repeatable)

        splitDSet = AudioDSet(dsName, samples=self.samples[splitIndexes], labels=self.labels[splitIndexes],
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
        """
        This class method creates a fine-tuning dataset and saves the information
        permanently on file system so that it can be loaded later.
        
        Parameters
        ----------
        dataPath : str
            The path to the directory where the dataset files are located.

        ratio : float
            The ratio between the number of samples in the fine-tuning dataset
            and total training samples.
            The default value of 0.1 results in creating a new dataset with %10 of the
            training samples.
        
        Notes
        -----
        This method can be used when consistent results is required. The same dataset
        samples are used every time the fine-tuning algorithm uses the dataset created
        by this method.
        """

        if dataPath is None:
            dataPath = '/data/AudioDb2/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
        
        dataset = np.load(dataPath + 'TrainDb.npz')
        trainSamples = dataset['samples']
        trainLabels = dataset['labels']
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
        This class method creates several datasets as specified by `dsNames` parameter
        in one-shot.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
                - "Train": Create the training dataset.
                - "Test":  Create the test dataset.
                - "Valid": Create the validation dataset. This uses the file
                           "ValidationDb.npz" to read samples and labels.
                           Unlike other datasets, for this data set,  the
                           validation samples are not extracted from the training
                           samples.
                - "Eval":  Create the evaluation dataset. This uses the file
                           "EvalDb.npz" to read samples and labels.
                - "Tune":  Create the Fine-Tuning datas. The ratio of fine-tuning
                           samples can be specified using a % sign followed by
                           the percentage. For example "Tune%5" means create
                           a fine-tuning dataset with %5 of the training datas.
                           If a percentage is not specified and a Tuning dataset
                           file (created by `createFineTuneDataset` function) is
                           available, the fine-tuning samples are loaded from the
                           existing file.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        Returns
        -------
        A list of `AudioDSet` objects
            Depending on the number of items specified in the `dsNames`, it returns a list
            of `AudioDSet` objects. The returned values have the same order as they
            appear in the `dsNames` parameter.

        Notes
        -----
        To specify the training dataset, any string containing the word "train" (case
        insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        To specify the test dataset, any string containing the word "test" (case
        insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        To specify the validation dataset, any string containing the word "valid" (case
        insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.
        To specify the evaluation dataset, any string containing the word "eval" (case
        insensitive) is accepted. So, "Evaluation", "EVAL", and 'eval' all can be used.
        To specify the fine-tuning dataset, any string containing the word "tun" (case
        insensitive) is accepted. So, "Fine-Tuning", "Tuning", and 'tune' all can be used.

        When the '%' is used to specify the ratio for 'Tuning' datasets, the subsampling is
        deterministic and the results are repeatable across different executions and even
        different platforms. If you want the results to be random, you can use '%r' instead of
        '%'. For example "Tune%r10" creates a dataset with %10 of training data which are
        selected randomly. A different call on the same or different machine will probably
        choose a different set of samples.
        
        Examples
        --------
        dsNames="Train,Test,Valid":
            3 `AudioDSet` objects are returned for training, test, and validation in
            the same order.

        dsNames="Train,Test":
            2 `AudioDSet` objects are returned for training and test.
            
        dsNames="FineTuning%r5,Test":
            2 `AudioDSet` objects are returned for fine-tuning and test. The fine-tuning
            dataset contains %5 of the training data (picked randomly because of '%r')
        
        dsNames="Tune%5,Test,Validation":
            3 `AudioDSet` objects are returned for fine-tuning, test, and validation in
            the same order. The fine-tuning and validation together contain %5 of available
            training data.
        """
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            retVals += [ cls(dsStr, dataPath, batchSize=batchSize) ]
            
        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals
