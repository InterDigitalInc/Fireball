# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of :py:class:`RadioMlDSet` class that encapsulates the `RadioML <https://www.deepsig.ai/datasets>`_ dataset for Modulation Classification problem.

The directory structure should be something like this::

    data
        RadioML
            RML2016_10b
            RML2018_01
            
Use the ``RadioMlDSetUnitTest.py`` file in the ``UnitTest/Datasets`` folder to run the Unit Test of this implementation.

The dataset contains samples of shape 128x1x2 (1024x1x2 for 2018 version) and labels which indicate one of 10 modulation types (24 for 2018 version). Each sample contains the floating point numbers captured as a time series. The time-series are the samples taken from I and Q signals. The samples were captured at 20 different SNR values (26 for 2018 version).

Note
----
    For the 2018 version, you need to download the "RML2018_01" dataset and run the :py:meth:`~fireball.datasets.radioml.RadioMlDSet.createNpzFiles` class method (only once) to extract the numpy files for each SNR value from the original "GOLD_XYZ_OSC.0001_1024.hdf5" file.

**Dataset Stats (2016)**
    
    * SNR Values: -20, -18, ... 18 (20 values)
    * Classes::
    
        0:8PSK      1:AM-DSB    2:BPSK      3:CPFSK     4:GFSK
        5:PAM4      6:QAM16     7:QAM64     8:QPSK      9:WBFM
        
    * Samples per SNR per class: 6,000
    * Samples per SNR value: 60,000
    * Samples per class: 120,000
    * Total Samples: 1,200,000 (= 10 * 20 * 6000)
    * The range of values in the dataset depends on the SNR values. The range is usually larger for larger SNR values; except for very small SNR values since they mostly contain noise. Here are the ranges for a few examples::
    
        SNR = -20: -0.030309 .. 0.032651
        SNR = -10: -0.030112 .. 0.031982
        SNR = 0:   -0.061123 .. 0.066305
        SNR = 10:  -0.129547 .. 0.106047
        SNR = 18:  -0.161878 .. 0.180627
        SNR = All: -0.210572 .. 0.180627

**Dataset Stats (2018)**

    * SNR Values: -20, -18, ... 28, 30 (26 values)
    * Classes::
    
        0:32PSK         1:16APSK        2:32QAM         3:FM
        4:GMSK          5:32APSK        6:OQPSK         7:8ASK
        8:BPSK          9:8PSK          10:AM-SSB-SC    11:4ASK
        12:16PSK        13:64APSK       14:128QAM       15:128APSK
        16:AM-DSB-SC    17:AM-SSB-WC    18:64QAM        19:QPSK
        20:256QAM       21:AM-DSB-WC    22:OOK          23:16QAM
    
    * Samples per SNR per class: 4,096
    * Samples per SNR value: 98,304
    * Samples per class: 106,496
    * Total Samples: 2,555,904  (= 24 * 26 * 4096)
    * Max Absolute Value in the whole dataset: 68.32339
    * The range of values in the dataset depends on the SNR values. The range is usually larger for larger SNR values; except for very small SNR values since they mostly contain noise. Here are the ranges for a few examples::
    
        SNR = -20: -3.921285 .. 3.855273
        SNR = -10: -3.941709 .. 4.169807
        SNR = 0:   -4.842472 .. 4.522851
        SNR = 10:  -12.566387 .. 12.534958
        SNR = 20:  -49.853298 .. 50.442608
        SNR = 30:  -37.581909 .. 41.837891
        SNR = All: -68.323387 .. 51.562645
"""

# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 03/03/2020              Shahab Hamidi-Rad       Created the file.
# 04/16/2020              Shahab Hamidi-Rad       Changed the way datasets are created. Added support
#                                                 for Fine-Tuning datasets, merging, splitting, and
#                                                 creating permanent fine-tune datasets.
#                                                 Also added the "createNpzFiles" class method to create
#                                                 the NPZ dataset files from the big original h5 file.
# 04/17/2020              Shahab Hamidi-Rad       Completed the documentation.
# 09/09/2020              Shahab Hamidi-Rad       Added support for the 2016 version. Re-structured the
#                                                 whole dataset code. Use "configure" to set the ratios
#                                                 for number of dataset samples.
# **********************************************************************************************************************
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint

# **********************************************************************************************************************
def randomSplit(whole, selection):
    if type(whole) not in (list, np.ndarray):   whole = range(whole)
    x = sorted(np.random.choice(whole, size=selection, replace=False))
    y = np.setdiff1d(whole, x).tolist()
    return x,y

# **********************************************************************************************************************
class RadioMlDSet(BaseDSet):
    """
    This class implements the RadioML dataset.
    """
    # These are the classes for the 2018 version of the dataset. This will be overwritten based on version
    # and labelMode values.
    classNames = None
    numClasses = 0
    
    version = None
    trainIndexes, testIndexes, validIndexes, tuneIndexes, tuneValidIndexes = None, None, None, None, None

    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, samples=None, labels=None, batchSize=128, snrValues=None,
                 version=2016, labelMode="MOD"):
        r"""
        Constructs a :py:class:`RadioMlDSet` instance. This can be called directly or via :py:meth:`makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. It can be one of "Train", "Test", or "Tune". Note that "Valid" cannot be used here.

        dataPath : str
            The path to the directory where the dataset files are located.

        samples : numpy array or None
            If specified, it is used as the samples for the dataset. It is a numpy array of samples. Each sample is numpy array of shape (128,1,2) for the 2016 version or (1024,1,2) for 2018 version.
            
        labels : numpy array or None
            If specified, it is a numpy array of int32 values. Each label is an int32 number indicating the class for each sample. (see the classNames above)
            
        batchSize : int
            The default batch size used in the "batches" method.
            
        snrValues : int, list of ints, or None
        
            * If it is an int, it specifies the single SNR value that must be used. The dataset will contain the data for the specified SNR value only.
            * If it is a list of ints, only the samples for the specified SNR values are included in the dataset.
            * If it is None, all samples for all SNR values are included in the dataset.
            
        version : int
            The version of the RadioML dataset. It can be either 1016 (the default) or 2018. The actual dataset versions used are as follows:
            
                * 2016: RADIOML 2016.10B
                * 2018: RADIOML 2018.01A

        labelMode : str
            Specifies the type of label to be returned in batches of the data.
            
                * ``MOD``: This mode returns the modulation class as the label. (Default)
                * ``SNR``: This mode returns the SNR index as the label
                * ``BOTH``: This mode returns a tuple of Modulation classes and SNR indexes as labels
        """
        
        if RadioMlDSet.version is None:    RadioMlDSet.configure(version=version)
        
        if samples is None: # If samples and labels are given, we don't need dataPath and it can be None.
            if dataPath is None:
                dataPath = '/data/RadioML/%s/'%('RML2016_10b' if version==2016 else 'RML2018_01')
                if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
            assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
            
        allSnrs = list(range(-20, 20, 2)) if version==2016 else list(range(-20, 32, 2))
        if snrValues is None:
            snrValues = allSnrs
        elif type(snrValues)==int:
            assert snrValues in allSnrs, "Invalid SNR value '%d'"%(snrValues)
            snrValues = np.int32([snrValues])
        else: # List / array
            for s in snrValues:
                assert s in allSnrs, "Invalid SNR value '%d'"%(s)
            snrValues = np.int32(snrValues)
        self.snrValues = snrValues

        self.snrs = None
        self.labelMode = labelMode # Options are: "MOD", "SNR", "BOTH"
        super().__init__(dsName, dataPath, samples, labels, batchSize)

    # ******************************************************************************************************************
    @classmethod
    def download(cls, dataFolder=None):
        r"""
        This class method can be called to download the RadioML dataset files from a Fireball
        online repository.
        
        Parameters
        ----------
        dataFolder: str
            The folder where dataset files are saved. If this is not provided, then
            a folder named "data" is created in the home directory of the current user and the
            dataset folders and files are created there. In other words, the default data folder
            is ``~/data``
        """
        
        BaseDSet.download("RadioML", ['RML2016_10b.zip'], dataFolder)

    # ******************************************************************************************************************
    @classmethod
    def configure(cls, testRatio=.5, tuneRatio=0.1, validRatio=0.1, version=2016):
        r"""
        Configures the :py:class:`RadioMlDSet` class. If a behavior other than the default is needed, this function can be called to prepare the class before instantiating the dataset instances.
        
        Since the RadioML dataset doesn't have a standard split for train, validation, and test samples, you can use this function to define the splits.
        
        Parameters
        ----------
        testRatio : float
            The ratio between the number of test samples to the total samples in the dataset. The default is 0.5 which means 50% of the data is used for training and 50% for test.
            
        tuneRatio : float
            The ratio between the number of tuning samples to the number of training sample. The default is 0.1, which means 10% of the training samples are used for tuning dataset.
        
        validRatio : float
            The ratio between the number of validation samples to the number of training sample. The default is 0.1, which means 10% of the training samples are used for validation dataset. The remaining 90% are used as training samples. Please note that if "valid" is not specified when :py:meth:`makeDatasets` is called, then all training samples are used for training.
        
        version : int
            The version of the RadioML dataset. It can be either 1016 (the default) or 2018. The actual dataset versions used are as follows:
            
                * 2016: RADIOML 2016.10B
                * 2018: RADIOML 2018.01A
        
        Example
        -------
            For example the following call::
            
                RadioMlDSet.configure(testRatio=.2, validRatio=0.1)
                
            can be used to have 20% of the samples for test, 72% for training, and 8% for validation.
            
            To Fine-Tune the trained model the following call::
            
                RadioMlDSet.configure(testRatio=.2, tuneRatio=0.1, validRatio=0.1)
                
            can be used to have 20% of samples for test, 7.2% for training (Fine-Tuning), and 0.8% for validation.
        """
        cls.version = version
        samplesPerSnrPerClass = 6000 if version==2016 else 4096
        testSamplesPerSnrPerClass = int(np.round(samplesPerSnrPerClass * testRatio))
        trainSamplesPerSnrPerClass = samplesPerSnrPerClass - testSamplesPerSnrPerClass
        tuneSamplesPerSnrPerClass = int(np.round(trainSamplesPerSnrPerClass * tuneRatio))
        validSamplesPerSnrPerClass = int(np.round(trainSamplesPerSnrPerClass * validRatio))
        trainSamplesPerSnrPerClass -= validSamplesPerSnrPerClass
        
        np.random.seed(version)
        cls.testIndexes, nonTestIndexes = randomSplit( samplesPerSnrPerClass, testSamplesPerSnrPerClass)
        cls.trainIndexes, cls.validIndexes = randomSplit( nonTestIndexes, trainSamplesPerSnrPerClass)
        tune, _ = randomSplit( nonTestIndexes, tuneSamplesPerSnrPerClass)
        tuneSamplesPerSnrPerClass -= int(np.round(tuneSamplesPerSnrPerClass * validRatio))
        cls.tuneIndexes, cls.tuneValidIndexes = randomSplit( tune, tuneSamplesPerSnrPerClass)
    
    # ******************************************************************************************************************
    def keepSnrs(self, snrsToKeep):
        assert self.version == 2016, "The 'keepSnrs' function only works for the 2016 version of the dataset!"
        assert len(self.snrValues) == 20, "The 'keepSnrs' function only works when all SNRs are available in this dataset!"
        assert self.labelMode == "MOD", "The 'keepSnrs' function only works in 'MOD' label mode!"
        indexesToKeep = np.concatenate([ np.where(self.snrs==((snr/2)+10))[0] for snr in snrsToKeep ])
        self.samples = self.samples[indexesToKeep]
        self.labels = self.labels[indexesToKeep]

        self.snrs = self.snrs[indexesToKeep]
        snrOldToNew = np.int32([1000]*20)
        for s,snr in enumerate(snrsToKeep): snrOldToNew[ int((snr/2)+10) ] = s
        self.snrs = snrOldToNew[self.snrs]
        assert 1000 not in self.snrs

        self.snrValues = np.int32(snrsToKeep)
        self.numSamples = len(self.samples)
        self.sampleIndexes = np.arange(self.numSamples)
    
    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This function is called by the constructor of the base dataset class to load the samples and labels of this dataset based on `dsName` and `dataPath` properties of this class.
        
        Note
        ----
        The dsName "Valid" cannot be used here. A validation dataset should be created using the `makeDatasets` method or using the `split` method on an existing training dataset.
        """
        allSamples = self.samples # This is set when the dataset is created by the makeDatasets.
        if allSamples is None:  allSamples = self.getAllSamples(self.dataPath, self.snrValues, self.version, self.labelMode)

        samplesPerSnrPerClass = 6000 if self.version==2016 else 4096
        numSnrs = len(self.snrValues)
        numMod = 10 if self.version==2016 else 24
        
        numAllSamples = samplesPerSnrPerClass * numMod * numSnrs
        assert numAllSamples == len(allSamples), "Total number of samples do not match!!! (%d vs %d)"%(numAllSamples,
                                                                                                       len(allSamples))

        allLabels = []
        allSnrs = []
        if self.version==2016:
            for c in range(numMod):
                for s in range(numSnrs):
                    allLabels += [ c ] * samplesPerSnrPerClass
                    allSnrs += [ s ] * samplesPerSnrPerClass
        else:
            for s in range(numSnrs):
                allLabels += [ c for c in range(numMod) for _ in range(samplesPerSnrPerClass) ]
                allSnrs += [ s ] * (samplesPerSnrPerClass*numMod)

        if self.version==2016 and (len(self.testIndexes)==samplesPerSnrPerClass//2) and numSnrs==20:
            # Special Case:
            # Running in version 2016, train/test ratio is .5, and all snrs are used.
            # In this case we want to handle train/test samples differently so that they
            # match this paper: "Fast Deep Learning for Automatic Modulation Classification"
            #   (https://arxiv.org/pdf/1901.05850.pdf)
            # Note that in this case samples are not balanced for classes and SNRs
            
            np.random.seed(2016)     # Random seed value for the partitioning
            trainIndexes, testIndexes = randomSplit(numAllSamples, numAllSamples//2)
            
            classSnrs = numMod * numSnrs
            totalTune = len(self.tuneIndexes) + len(self.tuneValidIndexes)
            tune, _ = randomSplit(trainIndexes, totalTune * classSnrs)
            tuneIndexes, tuneValidIndexes = randomSplit(tune, len(self.tuneIndexes) * classSnrs)
            trainIndexes, validIndexes = randomSplit(trainIndexes, len(self.trainIndexes) * classSnrs)
                                                     
            if 'train' in self.dsName.lower():
                dsIndexes = trainIndexes
                if self.dsName[-1] != 'V':      dsIndexes += validIndexes       # No valid => Use valid samples as train
                                        
            if 'test' in self.dsName.lower():   dsIndexes = testIndexes
                
            if 'valid' in self.dsName.lower():
                if self.dsName[-1] == 'T':      dsIndexes = tuneValidIndexes    # Validation for Tuning
                else:                           dsIndexes = validIndexes        # Validation for training
                    
            if 'tun' in self.dsName.lower():
                dsIndexes = tuneIndexes
                if self.dsName[-1] != 'V':      dsIndexes += tuneValidIndexes   # No valid => Use valid samples as tune
                    
        else:
            # allSamples:
            # 2016: (snr[0],class0),..., (snr[1],class0),...,...,(snr[n],class0),..., (snr[0],class1),...
            # 2018: (snr[0],class0),..., (snr[0],class1),...,...,(snr[0],class23),..., (snr[1],class0),...
            dsIndexes = []
            for c in range(numMod):
                for s in range(numSnrs):
                    if self.version == 2016:    start = (c*numSnrs + s)*samplesPerSnrPerClass
                    else:                       start = (s*numMod + c)*samplesPerSnrPerClass
                    if 'train' in self.dsName.lower():
                        dsIndexes += [x+start for x in self.trainIndexes]
                        if self.dsName[-1] != 'V':      dsIndexes += [x + start for x in self.validIndexes]
                            
                    if 'test' in self.dsName.lower():   dsIndexes += [x + start for x in self.testIndexes]
                        
                    if 'valid' in self.dsName.lower():
                        if self.dsName[-1] == 'T':      dsIndexes += [x + start for x in self.tuneValidIndexes]
                        else:                           dsIndexes += [x + start for x in self.validIndexes]
                            
                    if 'tun' in self.dsName.lower():
                        dsIndexes += [x + start for x in self.tuneIndexes]
                        if self.dsName[-1] != 'V':      dsIndexes += [x + start for x in self.tuneValidIndexes]
        
        if self.dsName[-1] in ['V', 'T']:   self.dsName = self.dsName[:-1] # Fix name (remove the temporary 'V'/'T')
        self.samples = allSamples[dsIndexes]
        self.labels = np.int32(allLabels)[dsIndexes]
        self.snrs = np.int32(allSnrs)[dsIndexes]

    # ******************************************************************************************************************
    @classmethod
    def getAllSamples(cls, dataPath, snrValues, version, labelMode):
        if version == 2016:
            import pickle
            Xd = pickle.load(open(dataPath + "RML2016.10b.dat",'rb'), encoding="latin1")
            snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
 
            if labelMode == "MOD":
                cls.classNames = mods
                cls.numClasses = len(cls.classNames)
            elif labelMode == "SNR":
                cls.classNames = [str(s) for s in snrValues]
                cls.numClasses = len(cls.classNames)
            else:
                cls.classNames = ( mods, [str(s) for s in snrValues] )
                cls.numClasses = ( len(cls.classNames[0]), len(cls.classNames[1]) )
        
            allSamples = []
            for mod in mods:
                for snr in snrValues:
                    allSamples += [ np.transpose(Xd[(mod,snr)], (0,2,1)).reshape((-1,128,1,2)) ]
            allSamples = np.concatenate(allSamples, axis=0)
            
        else:
            if labelMode == "MOD":
                cls.classNames = ['32PSK',      '16APSK',       '32QAM',        'FM',
                                  'GMSK',       '32APSK',       'OQPSK',        '8ASK',
                                  'BPSK',       '8PSK',         'AM-SSB-SC',    '4ASK',
                                  '16PSK',      '64APSK',       '128QAM',       '128APSK',
                                  'AM-DSB-SC',  'AM-SSB-WC',    '64QAM',        'QPSK',
                                  '256QAM',     'AM-DSB-WC',    'OOK',          '16QAM']
                cls.numClasses = len(cls.classNames)
            elif labelMode == "SNR":
                cls.classNames = [str(s) for s in snrValues]
                cls.numClasses = len(cls.classNames)
            elif labelMode == "BOTH":
                cls.classNames = ( cls.classNames, [str(s) for s in snrValues] )
                cls.numClasses = ( len(cls.classNames[0]), len(cls.classNames[1]) )

            allSamples = []
            for s, snrValue in enumerate(snrValues):
                npzFileName = dataPath + 'DataForSnr%s%02d.npz'%('-' if snrValue<0 else '+', abs(snrValue))
                dataDic = np.load(npzFileName, allow_pickle=True)
                assert dataDic["NumClasses"]==24
                assert dataDic["SamplesPerClass"]==4096
                assert dataDic["SNR"]==snrValue
                allSamples += [ dataDic["Samples"].reshape((-1, 1024, 1, 2)) ]
            allSamples = np.concatenate(allSamples, axis=0)
            
        return allSamples
        
    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=128, dataPath=None, snrValues=None, version=2016, labelMode="MOD"):
        r"""
        This class method creates several datasets as specified by `dsNames` parameter in one-shot.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
            
            * **"Train"**: Create the training dataset.
            * **"Test"**: Create the test dataset.
            * **"Valid"**: Create the validation dataset. The ratio of validation samples can be specified using the :py:meth:`configure` method before calling to this function. The default is 10% of training data. If it is used with tuning data, the ratio specifies the portion of tune samples (not train samples).
            * **"Tune"**: Create the Fine-Tuning datas. The ratio of fine-tuning samples can be specified using the :py:meth:`configure` method before calling to this function. The default is 10% of training data.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        snrValues : int, list of ints, or None
            Indicates which SNR values must be included in the dataset. Refer to the `snrValues` parameter of :py:meth:`__init__` method for more info.
                        
        version : int
            The version of the RadioML dataset. It can be either 1016 (the default) or 2018. The actual datasets used are as follows:
            
                * 2016: RADIOML 2016.10B
                * 2018: RADIOML 2018.01A
                
        labelMode : str
            Specifies the type of label to be returned in batches of the data.
            
                * ``MOD``: This mode returns the modulation class as the label. (Default)
                * ``SNR``: This mode returns the SNR index as the label
                * ``BOTH``: This mode returns a tuple of Modulation classes and SNR indexes as labels

        Returns
        -------
        Up to 3 :py:class:`RadioMlDSet` objects
            Depending on the number of items specified in the `dsNames`, it returns between one and three :py:class:`RadioMlDSet` objects. The returned values have the same order as they appear in the `dsNames` parameter.

        Note
        ----
        * To specify the training dataset, any string containing the word "train" (case insensitive) is accepted. So, "Training", "TRAIN", and 'train' all can be used.
        * To specify the test dataset, any string containing the word "test" (case insensitive) is accepted. So, "testing", "TEST", and 'test' all can be used.
        * To specify the validation dataset, any string containing the word "valid" (case insensitive) is accepted. So, "Validation", "VALID", and 'valid' all can be used.
        * To specify the fine-tuning dataset, any string containing the word "tun" (case insensitive) is accepted. So, "Fine-Tuning", "Tuning", and 'tune' all can be used.
        """

        if cls.version is None:    cls.configure(version=version)

        allSnrs = list(range(-20, 20, 2)) if version==2016 else list(range(-20, 32, 2))
        if snrValues is None:
            snrValues = allSnrs
        elif type(snrValues)==int:
            assert snrValues in allSnrs, "Invalid SNR value '%d'"%(snrValues)
            snrValues = np.int32([snrValues])
        else: # List / array
            for s in snrValues:
                assert s in allSnrs, "Invalid SNR value '%d'"%(s)
            snrValues = np.int32(snrValues)

        if dataPath is None:
            dataPath = '/data/RadioML/%s/'%('RML2016_10b' if version==2016 else 'RML2018_01')
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        allSamples = cls.getAllSamples(dataPath, snrValues, version, labelMode)
        
        validSource = None
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            if ('train' in dsStr.lower()) or ('tun' in dsStr.lower()):
                tempDsName = (dsStr + 'V') if 'valid' in dsNames.lower() else dsStr
                retVals += [ cls(tempDsName, dataPath, samples=allSamples, batchSize=batchSize,
                                 snrValues=snrValues, version=version, labelMode=labelMode) ]
                            
            elif 'valid' in dsStr.lower():
                tempDsName = (dsStr + 'T') if 'tun' in dsNames.lower() else dsStr
                retVals += [ cls(tempDsName, dataPath, samples=allSamples, batchSize=batchSize,
                                 snrValues=snrValues, version=version, labelMode=labelMode) ]
                
            else:
                retVals += [ cls(dsStr, dataPath, samples=allSamples, batchSize=batchSize,
                                 snrValues=snrValues, version=version, labelMode=labelMode) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals

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
        numAllSnrs = 20 if self.version == 2016 else 26
        snrValusesStr = 'All' if len(self.snrValues)==numAllSnrs else str(self.snrValues)
        repStr += '    Version ........................................ %d\n'%(self.version)
        repStr += '    SNR Values ..................................... %s\n'%(snrValusesStr)
        repStr += '    Label Mode ..................................... %s\n'%(self.labelMode)
        return repStr
    # ******************************************************************************************************************
    @classmethod
    def printDsInfo(cls, trainDs=None, testDs=None, validDs=None):
        r"""
        This class method prints information about given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : any object derived from `BaseDSet`, optional
            The training dataset.
            
        testDs : any object derived from `BaseDSet`, optional
            The test dataset.
                
        validDs : any object derived from `BaseDSet`, optional
            The validation dataset.
        """
        
        def commonInfo(x):  return x.dataPath, x.sampleShape, x.snrValues, x.labelMode, x.version
        if trainDs is not None:     dataPath, sampleShape, snrValues, labelMode, version = commonInfo(trainDs)
        elif testDs is not None:    dataPath, sampleShape, snrValues, labelMode, version = commonInfo(testDs)
        elif validDs is not None:   dataPath, sampleShape, snrValues, labelMode, version = commonInfo(validDs)

        numAllSnrs = 20 if version == 2016 else 26
        snrValusesStr = 'All (%d)'%(numAllSnrs) if len(snrValues)==numAllSnrs else str(snrValues)

        print('%s Dataset Info:'%(cls.__name__))
        print('    Version ........................................ %d'%(version))
        print('    SNR Values ..................................... %s'%(snrValusesStr))
        print('    Label Mode ..................................... %s'%(labelMode))
        print('    Dataset Location ............................... %s'%(dataPath))
        print('    Number of Classes .............................. %s'%(str(cls.numClasses)))
        if trainDs is not None:
            print('    Number of Training Samples ..................... %d'%(trainDs.numSamples))
        if testDs is not None:
            print('    Number of Test Samples ......................... %d'%(testDs.numSamples))
        if validDs is not None:
            print('    Number of Validation Samples ................... %d'%(validDs.numSamples))
        print('    Sample Shape ................................... %s'%(str(sampleShape)))

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
        samples : list or numpy array
            The batch samples specified by the `batchIndexes`.
            
        labels : list or numpy array
            The batch labels specified by the `batchIndexes`.
        """
        if self.labelMode == "MOD":     # Labels are modulation classes
            return self.samples[batchIndexes], self.labels[batchIndexes]
        
        if self.labelMode == "SNR":     # Labels are SNR classes
            return self.samples[batchIndexes], self.snrs[batchIndexes]
        
        # Labels are tuples of modulation and SNR classes:
        return self.samples[batchIndexes], (self.labels[batchIndexes], self.snrs[batchIndexes])

    # ******************************************************************************************************************
    @classmethod
    def createNpzFiles(cls, dataPath=None):
        r"""
        For the 2018 version of RadioML dataset, this function reads the dataset information from the original dataset file ``GOLD_XYZ_OSC.0001_1024.hdf5`` and creates an "npz" file for each SNR value.

        This only needs to be done once before this dataset can be used.

        Parameters
        ----------
        dataPath : str
            The path to the directory where the dataset files are located.
        """
        
        import h5py
        if dataPath is None:
            dataPath = '/data/RadioML/RML2018_01/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        print('Loading the dataset from "%sGOLD_XYZ_OSC.0001_1024.hdf5"...'%(dataPath))
        h5File = h5py.File(dataPath + "GOLD_XYZ_OSC.0001_1024.hdf5", mode='r')

        allSamples = np.array(h5File['X'])
        allLabels = np.argmax(np.array(h5File['Y']), axis=1)
        allSnrs = np.array(h5File['Z'])

        snrVals = np.unique(allSnrs)
        labels = np.unique(allLabels)
        print('Loaded %d samples (at %d different SNR values, %d classes).'%(allSamples.shape[0],
                                                                             snrVals.shape[0], labels.shape[0]))
        for snr in snrVals:
            myPrint( 'Creating SNR dataset for SNR=%d ... '%(snr), False)
            snrIndexes = np.where(allSnrs==snr)[0]
            assert len(snrIndexes)==(24*4096)
            samples = allSamples[ snrIndexes ]
            labels = allLabels[ snrIndexes ]
            for x in range(24):
                assert labels[x*4096:(x+1)*4096].max() == labels[x*4096:(x+1)*4096].min(), "%d :: %d"%(labels[x*4096:(x+1)*4096].max(), labels[x*4096:(x+1)*4096].min())
                assert labels[x*4096:(x+1)*4096].max() == x, "%d :: %d"%(labels[x*4096:(x+1)*4096].max(),x)

            DataForSnrDic = { "Samples": samples, "SamplesPerClass": 4096, "SNR": snr, "NumClasses": 24 }
            fileName = dataPath + 'DataForSnr%s%02d.npz'%('-' if snr<0 else '+', abs(snr))
            np.savez_compressed(fileName, **DataForSnrDic)
            myPrint( 'Done.' )
