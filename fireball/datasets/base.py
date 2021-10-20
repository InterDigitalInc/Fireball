# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the implementation of the base dataset class for all other datasets.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed            By                      Description
# ------------            --------------------    ------------------------------------------------
# 03/03/2020              Shahab Hamidi-Rad       Created the file.
# 04/15/2020              Shahab Hamidi-Rad       Changed the way datasets are created. Added support
#                                                 for Fine-Tuning datasets, merging, and splitting
#                                                 datasets.
# 04/17/2020              Shahab Hamidi-Rad       Completed the documentation.
# 11/19/2020              Shahab Hamidi-Rad       Improved calculation top-K accuracy. Also added more
#                                                 arguments to control the behavior of the evaluation
#                                                 functions.
# 10/11/2021              Shahab Hamidi-Rad       Added support for downloading datasets.
# **********************************************************************************************************************
import numpy as np
import os
import time
from threading import Thread
from queue import Queue
import zipfile
import yaml
import urllib.request as urlreq

from ..printutils import myPrint

# **********************************************************************************************************************
def batchLoader(jobs, results):
    r"""
    This worker function runs in its own thread. It loads a batch of samples and labels and puts them in a queue ready to be processed by the model. This method gets the batch indexes for the next batch from the 'jobs' Queue, loads them by calling the `getBatch` method of the dataset object, and puts the results (the loaded batch of samples) in the `results` Queue.
    
    Parameters
    ----------
    jobs : Queue
        A Queue object containing information about the next batches waiting to be processed.

    results : Queue
        A Queue object containing the loaded batches that are ready to be processed.
    """
    while True:
        batchIndexes, dataset = jobs.get()
        if batchIndexes is None:    break
        batchSamples, batchLabels = dataset.getBatch(batchIndexes)
        results.put((batchSamples, batchLabels))

# **********************************************************************************************************************
# **********************************************************************************************************************
class BaseDSet:
    r"""
    The base class for all other Dataset classes defined in this folder. This class can be used as the base class for both classification and regression problems.
    
    Please refer to the mnist.py for an example of how to derive from this class.
    """
    
    # These are class properties. Do not set them for a single object. (self.numClasses = 2 is wrong!)
    classNames = None
    numClasses = 0
    psnrMax = None
    evalMetricName = None

    # ******************************************************************************************************************
    def __init__(self, dsName, dataPath, samples, labels, batchSize, numWorkers=0):
        r"""
        Constructs a BaseDSet class. Never called directly. It is always called from constructor of a derived class.
        
        Parameters
        ----------
        dsName : str
            The name of the dataset. Common names include "Train", "Test", and "Valid". Some derived classes may include additional application-specific names.
            
        dataPath : str
            The path to the directory where the dataset files are located.
            
        samples : list, numpy array, or None
        
            * If specified it is used as the samples for the dataset. Depending on the application it can be a list or a numpy array.
            * If samples is not specified, the loadSamples() method is called. This method MUST be implemented by any derived class.
            
        labels : list, numpy array, or None
            If specified it will be used as the labels for the dataset. Depending on the application it can be a list or a numpy array.
            
        batchSize : int
            The default batch size used in the "batches" method.
            
        numWorkers : int, optional
            If numWorkers is more than zero, “numWorkers” worker threads are created to process and prepare future batches in parallel.
        """
    
        self.dsName = dsName
        self.dataPath = dataPath
        self.samples = samples
        self.labels = labels
        self.numWorkers = numWorkers
        if (self.samples is None) or (self.labels is None): self.loadSamples()
        self.batchSize = batchSize
        self.numSamples = len(self.samples)
        self.sampleShape = self.samples.shape[1:] if type(self.samples)==np.ndarray else None
        self.labelShape = self.labels.shape[1:] if type(self.labels)==np.ndarray else None
        self.sampleIndexes = np.arange(self.numSamples)
        self.__class__.evalMetricName = 'Error' if self.numClasses>0 else 'MSE'

    # ******************************************************************************************************************
    @property
    def evalMetricBiggerIsBetter(self):
        r"""
        Returns True for the evaluation metric of this dataset, a larger value is better. The default implementation in this base class returns True for metrics **mAP**, **Accuracy**, and **PSNR** and False for other metrics.
        """
        return (self.evalMetricName.lower() in ['map', 'accuracy', 'psnr'])

    # ******************************************************************************************************************
    @property
    def isTraining(self):
        r"""
        Returns True if this dataset instance is a training dataset.
        """
        return ('train' in self.dsName.lower())

    # ******************************************************************************************************************
    def getLabelAt(self, idx):
        r"""
        A method to return the label for the sample specified by the `idx`. This function should be implemented by the derived classes if they have a special way of accessing labels. For an example see the implementation of this method in the `ImageNetDSet` class.
        
        Parameters
        ----------
        idx : int
            The index specifying the sample in the dataset.
            
        Returns
        -------
        int, float, or numpy array
            The label for the specified sample. For classification problems this is usually an integer specifying the class the sample belongs to. For regression problem this may be a single floating point value or a numpy array.
        """
        
        return self.labels[ idx ]
    
    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This method loads the dataset information from the dataset files. For the base class, this is just a place holder.
        
        Note
        ----
        This method MUST be implemented by the derived classes.
        """
        assert self.samples is not None, "The function 'loadSamples' must to be defined in the dataset class!"
        
    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test,Valid', batchSize=32, dataPath=None, printInfo=False):
        """
        This class method creates Train, Test, and/or Validation datasets. For the base class, this is just a place holder.

        Note
        ----
        This method MUST be implemented by the derived classes.
        """
        
        assert False, "The \"makeDatasets\" function must be implemented in the derived classes!"
 
     # ******************************************************************************************************************
    @classmethod
    def postMakeDatasets(cls):
        """
        This class method is called at the end of a call to makeDatasets. This can be used by derived classes to setup the dataset class after the datasets are created.
        """
        pass

    # ******************************************************************************************************************
    def split(self, dsName='Valid', ratio=.1, batchSize=None, repeatable=True):
        """
        This method must be implemented by the derived classes.
        """
        
        assert False, "The \"split\" function must be implemented in the derived classes!"

    # ******************************************************************************************************************
    def getSplitIndexes(self, ratio=.1, repeatable=True):
        r"""
        This function returns a list of sample indexes that can be used to split this dataset. This is a utility function usually used by the `split` function implemented by the derived classes. As an example, refer the implementation of the `split` function in the `MnistDSet` class.
                
        Parameters
        ----------
        ratio : float, optional
            The ratio of the number of split indexes to the total number of samples in this dataset. Default is 10 percent of the sample.
        
        repeatable : Boolean, optional
            If True, the sampling from the original dataset is deterministic and therefore the experiments are repeatable. Otherwise, the sampling is done randomly.
            
        Returns
        -------
        list
            A list of indexes of the samples included in the split.
        """
        
        numSplitSamples = int(np.round(self.numSamples*ratio))
        splitIndexes = []
        nSplit, nRemaining = 0, 0
        for c in range(self.numClasses):
            thisClassIndexes = np.where(self.labels==c)[0]
            nThisClass = len( thisClassIndexes )
            
            if c == (self.numClasses-1):
                nThisClassSplit = numSplitSamples - nSplit
            else:
                dynamicRatio = float(numSplitSamples-nSplit)/(self.numSamples-nSplit-nRemaining)
                nThisClassSplit = int(np.round(nThisClass*dynamicRatio))
            
            nSplit += nThisClassSplit
            nRemaining += nThisClass - nThisClassSplit
            
            if repeatable:
                splitIndexes += thisClassIndexes[:nThisClassSplit].tolist()
            else:
                splitIndexes += sorted(np.random.choice(thisClassIndexes, nThisClassSplit, replace=False))

        return splitIndexes

    # ******************************************************************************************************************
    def mergeWith(self, otherDSet):
        r"""
        This function merges the contents of this dataset with the contents of another dataset specified by `otherDSet`. The datasets must have similar properties. The dataset `otherDSet` remains unchanged.
        
        Parameters
        ----------
        otherDSet : any object derived from `BaseDSet`
            The contents of this dataset is merged with "self".
        """
        
        assert ((self.batchSize == otherDSet.batchSize) and (self.sampleShape == otherDSet.sampleShape)), \
               "Datasets can be merged only if all their properties match!"

        if type(self.samples)==np.ndarray:
            self.samples = np.concatenate([self.samples, otherDSet.samples], axis=0)
        else:
            self.samples += otherDSet.samples   # Assuming samples is a list
        
        if type(self.labels)==np.ndarray:
            self.labels = np.int32(self.labels.tolist() + otherDSet.labels.tolist())
        
        self.numSamples = len(self.samples)
        self.sampleIndexes = np.arange(self.numSamples)

    # ******************************************************************************************************************
    def __repr__(self):
        r"""
        Provides a text string containing the information about this object.
        
        Returns
        -------
        str
            The text string containing the information about this object.
        """
        
        repStr = '%s Dataset Info:\n'%(self.__class__.__name__)
        repStr += '    Dataset Name ................................... %s\n'%(self.dsName)
        if self.dataPath is not None:
            repStr += '    Dataset Location ............................... %s\n'%(self.dataPath)
        if self.numClasses>0:
            repStr += '    Number of Classes .............................. %d\n'%(self.numClasses)
        repStr += '    Number of Samples .............................. %d\n'%(self.numSamples)
        repStr += '    Sample Shape ................................... %s\n'%(str(self.sampleShape))
        if self.numClasses==0:
            repStr += '    Label Shape .................................... %s\n'%(str(self.labelShape))
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
        
        if trainDs is not None:     dataPath, sampleShape = trainDs.dataPath, trainDs.sampleShape
        elif testDs is not None:    dataPath, sampleShape = testDs.dataPath, testDs.sampleShape
        elif validDs is not None:   dataPath, sampleShape = validDs.dataPath, validDs.sampleShape

        print('%s Dataset Info:'%(cls.__name__))
        if cls.numClasses>0:
            print('    Number of Classes .............................. %d'%(cls.numClasses))
        if dataPath is not None:
            print('    Dataset Location ............................... %s'%(dataPath))
        if trainDs is not None:
            print('    Number of Training Samples ..................... %d'%(trainDs.numSamples))
        if testDs is not None:
            print('    Number of Test Samples ......................... %d'%(testDs.numSamples))
        if validDs is not None:
            print('    Number of Validation Samples ................... %d'%(validDs.numSamples))
        if sampleShape is not None:
            print('    Sample Shape ................................... %s'%(str(sampleShape)))

    # ******************************************************************************************************************
    @classmethod
    def printStats(cls, trainDs=None, testDs=None, validDs=None):
        r"""
        This class method prints statistics of classes for the given set of datasets in a single table. This is used only for the classification datasets.
        
        Parameters
        ----------
        trainDs : any object derived from `BaseDSet`, optional
            The training dataset.
            
        testDs : any object derived from `BaseDSet`, optional
                The test dataset.
                
        validDs : any object derived from `BaseDSet`, optional
                The validation dataset.
        """
        
        assert cls.numClasses>0, "'printStats' can only be called for a Classification dataset!"
        maxClassWidth = 0
        for className in cls.classNames:
            if len(className)>maxClassWidth:
                maxClassWidth = len(className)
        
        #             | 123 1234567890123 |
        sep =    '    +-----' + '-'*maxClassWidth + '-+'
        rowStr = '    | Class' + ' '*maxClassWidth + '|'
        
        all = 0
        if trainDs is not None:
            if trainDs.labels is None:  return
            trainCounts = np.zeros(cls.numClasses)
            for _,label in trainDs.batches(1): trainCounts[ label[0] ] += 1
            all += trainDs.numSamples
            #           1234567890123456 |
            sep +=    '------------------+'
            rowStr += ' Training Samples |'

        if validDs is not None:
            if validDs.labels is None:  return
            validCounts = np.zeros(cls.numClasses)
            for _,label in validDs.batches(1): validCounts[ label[0] ] += 1
            all += validDs.numSamples
            #           123456789012345678 |
            sep +=    '--------------------+'
            rowStr += ' Validation Samples |'

        if testDs is not None:
            if testDs.labels is None:   return
            testCounts = np.zeros(cls.numClasses)
            for _,label in testDs.batches(1): testCounts[ label[0] ] += 1
            all += testDs.numSamples
            #           1234567890123 |
            sep +=    '---------------+'
            rowStr += ' Test Samples  |'

        print(sep)
        print(rowStr)
        print(sep)

        for c in range(cls.numClasses):
            if str(c) == cls.classNames[c]: rowStr = ('    | %%-%dd |'%(4+maxClassWidth))%(c)    # Mostly for MNIST!
            else:                           rowStr = ('    | %%3d %%-%ds |'%(maxClassWidth))%(c, cls.classNames[c])
            if trainDs is not None:
                rowStr += ' %-9d %5.2f%% |'%(trainCounts[c], trainCounts[c]*100.0/trainDs.numSamples)
            if validDs is not None:
                rowStr += ' %-9d   %5.2f%% |'%(validCounts[c], validCounts[c]*100.0/validDs.numSamples)
            if testDs is not None:
                rowStr += ' %-6d %5.2f%% |'%(testCounts[c], testCounts[c]*100.0/testDs.numSamples)

            print( rowStr )
       
        #             | 123 1234567890123 |
        rowStr = '    | Total' + ' '*maxClassWidth + '|'
        if trainDs is not None:
            rowStr += ' %-8d %6.2f%% |' % (trainDs.numSamples, 100.0*trainDs.numSamples/all)
        if validDs is not None:
            rowStr += ' %-9d   %5.2f%% |'%(validDs.numSamples, 100.0*validDs.numSamples/all)
        if testDs is not None:
            rowStr += ' %-6d %5.2f%% |'%(testDs.numSamples, 100.0*testDs.numSamples/all)

        print(sep)
        print(rowStr)
        print(sep)

    # ******************************************************************************************************************
    def boostClass(self, classIndex, ratio):
        r"""
        This method increases the number of samples for the specified class by the specified ratio. For example if the ratio is 2, the dataset will have twice the original number of sample for the specified class.
        
        This can be used to add bias for a specific class for training. It is recommended to call this function just before calling the train function of the model.
        
        Parameters
        ----------
        classIndex : int
            A index of class. Must be in "range(self.numClasses)"
            
        ratio: float
            The ratio of the new number of samples for the specified class to the original number of samples for that class.
        """
        assert ratio > 1.0, "The 'ratio' must be greater than 1!"
        assert classIndex < self.numClasses, "Invalid 'classIndex'(%d). Valid range is 0 to %d"%(classIndex, self.numClasses-1)
        
        classSampleIndexes = np.where(self.labels==classIndex)[0].tolist()
        n = len(classSampleIndexes)
        classSampleIndexesAll = []
        if ratio>=2.0:
            classSampleIndexesAll = classSampleIndexes * int(ratio-1)
            
        ratio -= int(ratio)
        if (ratio * n)>1:
            classSampleIndexesAll += np.random.choice(classSampleIndexes, int(ratio*n), replace=False).tolist()
        
        self.sampleIndexes = np.int32( self.sampleIndexes.tolist() + classSampleIndexesAll )
        self.numSamples = len(self.sampleIndexes)

    # ******************************************************************************************************************
    def getBatch(self, batchIndexes):
        r"""
        This method returns a batch of samples and labels from the dataset as specified by the list of indexes in the `batchIndexes` parameter. This is a generic implementation that works for most cases. A derived class may implement a customized version of this method.
        
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
        return self.samples[batchIndexes], self.labels[batchIndexes]

    # ******************************************************************************************************************
    def batches(self, iterBatchSize=None, numWorkers=None, sampleIndexes=None):
        r"""
        This generator function is used to loop through all the samples. This is used by the `train` method of a `Model` object during the training. This can also be used when evaluating a model using validation or test datasets.
                
        If current dataset is a training or fine-tuning dataset, the samples are shuffled before the start of the loop.
        
        Parameters
        ----------
        iterBatchSize : int, optional
            If not specified the default batch size specified in the `__init__` is used. Otherwise this value overrides the predefine batchSize.
            
        numWorkers : int or None, optional
            If not specified (=None, default) the dataset's numWorkers is used. If numWorkers is more than zero, "numWorkers" worker threads are created to process and prepare future batches in parallel.
            
        sampleIndexes : list of int, optional
            if specified the batches are taken only from the samples specified by "sampleIndexes". Otherwise (default), all available samples are considered for the batches.
            
        Yields
        ------
        samples : numpy array
            The next batch of samples.

        labels : list of 2-tuples or a 3-tuple of numpy arrays
            The next batch of labels.
            
        Note
        ----
        * This function first obtains a list of indexes for the next batch of the dataset. If the number of workers is 0, it calls the `getBatch` function to return the batch samples and labels.
        * If number of workers is not zero, the batch indexes are put in the `jobs` Queue which are then used by one of the worker threads.
        """

        if numWorkers is None:    numWorkers = self.numWorkers
        if numWorkers>0:
            jobs = Queue()
            results = Queue(2*numWorkers)
            workers = []
            for _ in range(numWorkers):
                workers += [ Thread(target=batchLoader, args = (jobs, results), daemon=True) ]
                workers[-1].start()

        if sampleIndexes is None:   sampleIndexes = self.sampleIndexes
        numSamples = len(sampleIndexes)
        bs = self.batchSize if iterBatchSize is None else iterBatchSize
        numBatches = numSamples//bs
        if numBatches*bs < numSamples: numBatches += 1
        if self.isTraining or ('tun' in self.dsName.lower()): np.random.shuffle(sampleIndexes)

        j = 0
        for b in range(numBatches):
            batchIndexes = sampleIndexes[b*bs : (b+1)*bs]
            if numWorkers == 0:
                batchSamples, batchLabels = self.getBatch(batchIndexes)
            else:
                while jobs.qsize() < (2*numWorkers):
                    if j<numBatches:
                        batchIndexes = sampleIndexes[j*bs : (j+1)*bs]
                        jobs.put((batchIndexes, self))
                        j += 1
                    else:
                        # We are done. Signal the workers to quit
                        jobs.put((None, None))
            
                # All workers busy, wait for one of them to be done:
                batchSamples, batchLabels = results.get()
                
            yield batchSamples, batchLabels

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
                
                * **sampleIndexes (list of ints)**: A list of sample indexes from this dataset to be processed for the evaluation of the model. If not specified, all samples are used. (default behavior).

                * **topK (int)**: For classification cases, this indicates whether a "top-K" accuracy value should also be calculated. For example for ImageNet dataset classification, usually the top-5 accuracy value is used (topK=5) besides the top-1. If it is zero (default), the top-K error is not calculated. This is ignored for regression cases.

                * **confMat (Boolean)**: For classification cases, this indicates whether the confusion matrix should be calculated. If the number of classes is more than 10, this argument is ignored and confusion matrix is not calculated. This is ignored for regression cases.

                * **expAcc (Boolean or None)**: Ignored for regression cases. For classification cases:
                 
                    * If this is a True, the expected accuracy and kappa values are also calculated. When the number of classes and/or number of evaluation samples is large, calculating expected accuracy can take a long time.
                    * If this is False, the expected accuracy and kappa are not calculated.
                    * If this is None (the default), then the expected accuracy and kappa are calculated only if number of classes does not exceed 10.
                                        
                    **Note**: If confMat is True, then expAcc is automatically set to True.

                * **jsonFile (str)**: The name of JSON file that is created by this function. This is used with some NLP applications where the results could be saved to a JSON file for evaluation.
                    
        Returns
        -------
        If returnMetric is True, the actual value of dataset's main metric is returned.
        Otherwise, this function returns a dictionary containing the results of the evaluation process.
        """
        if model.layers.output.supportsEval:
            return self.evalMultiDimRegression(model, batchSize, quiet, returnMetric, **kwargs)
            
        maxSamples=kwargs.get('maxSamples', None)
        topK=kwargs.get('topK', 0)
        specifiedSamples=kwargs.get('sampleIndexes', None)
        expAcc = kwargs.get('expAcc', None)

        t0 = time.time()
        if maxSamples is None:  maxSamples = self.numSamples
        if batchSize is None:   batchSize = self.batchSize
        processQuiet = quiet or returnMetric
        totalSamples = 0

        predictions, actuals = [], []
        inferResults = []
        
        totalTime = 0 # If batchSize is 1, we want to calculate the average inference time per sample.
        for b, (batchSamples, batchLabels) in enumerate(self.batches(batchSize, sampleIndexes=specifiedSamples)):
            if totalSamples>=maxSamples: break
            if returnMetric and (not quiet):
                model.updateTrainingTable('  Running Inference for %s sample %d ... '%(self.dsName.lower(), totalSamples))
            if not processQuiet:
                if batchSize==1:
                    myPrint('\r  Processing sample %d ... '%(b+1), False)
                else:
                    myPrint('\r  Processing batch %d - (Total Samples so far: %d) ... '%(b+1, totalSamples), False)
            
            if type(batchSamples) == tuple:     totalSamples += len(batchSamples[0])
            else:                               totalSamples += len(batchSamples)
            if batchSize == 1: t0 = time.time()
            
            if type(batchLabels) == tuple:
                inferResults = model.inferBatch( batchSamples, returnProbs=False )
                tupLen = len(batchLabels)
                assert type(inferResults) == tuple, "\"batchLabels\" is a tuple but \"inferBatch\" did not " \
                                                    "returned a tuple!"
                assert len(inferResults) == tupLen, "\"batchLabels\" is %d-tuple but \"inferBatch\" " \
                                                    "returned a %d-tuple!"%(tupLen, len(inferResults))
                if len(predictions)==0:
                    # First time: Make predictions to be tuple of lists instead of list
                    predictions = tuple(predList.tolist() for predList in inferResults)
                    actuals = tuple(actList.tolist() for actList in batchLabels)
                else:
                    predictions = tuple( predictions[i]+inferResults[i].tolist() for i in range(tupLen) )
                    actuals = tuple( actuals[i]+batchLabels[i].tolist() for i in range(tupLen) )
                    
            # Normal cases (batchLabels is not a tuple)
            elif topK>0:
                predictions += np.argsort( model.inferBatch( batchSamples ) )[:,-topK:].tolist() # Keep topK indexes
                actuals += batchLabels.tolist()

            else:
                predictions += model.inferBatch( batchSamples, returnProbs=False).tolist()
                actuals += batchLabels.tolist()
            if b>0: totalTime +=(time.time()-t0)    # Do not count the first sample

        if type(actuals) != tuple:
            # If actuals and predictions are tuples, it is responsibility of the derived "getMetricVal" and
            # "evaluate" functions called bellow to handle them correctly.
            if self.numClasses == 0:    predictions, actuals = np.float32(predictions), np.float32(actuals)
            else:                       predictions, actuals = np.array(predictions), np.int32(actuals)

        if returnMetric:
            return self.getMetricVal(predictions, actuals)

        if not processQuiet:
            if batchSize==1:
                myPrint('\r  Processed %d Sample. (Time Per Sample: %.2f ms)%30s\n'%(totalSamples, (1000.0*totalTime)/(maxSamples-1),' '))
            else:
                myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)%30s\n'%(totalSamples, time.time()-t0,' '))

        evalResults = self.evaluate(predictions, actuals, topK, confMat=kwargs.get('confMat', False),
                                    expAcc=expAcc, quiet=processQuiet)
        
        if model.bestMetric is not None:
            evalResults['best%s'%(self.evalMetricName)] = model.bestMetric
            evalResults['bestEpoch'] = model.bestEpochInfo[0]+1
            evalResults['trainTime'] = model.trainTime
            evalResults['csvItems'] += ['best%s'%(self.evalMetricName),'bestEpoch','trainTime']

        return evalResults

    # ******************************************************************************************************************
    def evalMultiDimRegression(self, model, batchSize=None, quiet=False, returnMetric=False, **kwargs):
        r"""
        This function evaluates the specified model using this dataset. This is currently only used for multi-dimensional regression problems if the output layer supports evaluation in the graph. (supportsEval is true)
        
        The list of parameters and Return values for this function is the same as the "evaluateModel" function defined above.
        """
        maxSamples=kwargs.get('maxSamples', None)
        topK=kwargs.get('topK', 0)
        
        t0 = time.time()
        if maxSamples is None:  maxSamples = self.numSamples
        if batchSize is None:   batchSize = self.batchSize
        processQuiet = quiet or returnMetric
        totalSamples = 0

        sumSquaredErrors, sumAbsoluteErrors = [], []
        labelSize = None
        totalTime = 0 # If batchSize is 1, we want to calculate the average inference time per sample.
        for b, (batchSamples, batchLabels) in enumerate(self.batches(batchSize)):
            if labelSize is None: labelSize = np.prod(batchLabels.shape[1:])
            if totalSamples>=maxSamples: break
            if returnMetric and (not quiet):
                model.updateTrainingTable('  Running Evaluation for %s sample %d ... '%(self.dsName.lower(), totalSamples))
            if not processQuiet:
                if batchSize==1:
                    myPrint('\r  Processing sample %d ... '%(b+1), False)
                else:
                    myPrint('\r  Processing batch %d - (Total Samples so far: %d) ... '%(b+1, totalSamples), False)
            
            if type(batchSamples) == tuple:     totalSamples += len(batchSamples[0])
            else:                               totalSamples += len(batchSamples)
            if batchSize == 1: t0 = time.time()
            
            batchSSE, batchSAE = model.evalBatch(batchSamples, batchLabels)
            sumSquaredErrors += batchSSE.tolist()
            sumAbsoluteErrors += batchSAE.tolist()

            if b>0: totalTime +=(time.time()-t0)    # Do not count the first sample

        # Calculating average of MSE, RMSE, MAE, PSNR
        # Calculate MSE, RMSE, MAE, PSNR for each sample and then return the average over all samples.
        mses = np.float32(sumSquaredErrors)/labelSize
        rmses = np.sqrt(mses)
        rmsesClipped = np.clip(rmses, 0.000001, None)
        maes = np.float32(sumAbsoluteErrors)/labelSize
        
        aMse = mses.mean()
        aRmse = rmses.mean()
        aMae = maes.mean()
        aPsnr = None if self.psnrMax is None else (20.*np.log10(self.psnrMax/rmsesClipped)).mean()

        # Calculating global MSE, RMSE, MAE, PSNR
        # Calculating MSE, RMSE, MAE, PSNR over the whole data at once.
        gMse = np.float32(sumSquaredErrors).sum()/(labelSize*maxSamples)
        gMae = np.float32(sumAbsoluteErrors).sum()/(labelSize*maxSamples)
        gRmse = np.sqrt(gMse)
        rmsesClipped = np.clip(gRmse, 0.000001, None)
        gPsnr = None if self.psnrMax is None else (20.*np.log10(self.psnrMax/rmsesClipped))

        results = {
            'mse':          aMse,
            'rmse':         aRmse,
            'mae':          aMae,
            'gMse':         gMse,
            'gRmse':        gRmse,
            'gMae':         gMae,

            'csvItems':     ['mse','rmse','mae'],
        }
        if aPsnr is not None:
            results['psnr'] = aPsnr
            results['csvItems'] += ['psnr']
        if gPsnr is not None:
            results['gPsnr'] = gPsnr

        if model.bestMetric is not None:
            results['best%s'%(self.evalMetricName)] = model.bestMetric
            results['bestEpoch'] = model.bestEpochInfo[0]+1
            results['trainTime'] = model.trainTime
            results['csvItems'] += ['best%s'%(self.evalMetricName),'bestEpoch','trainTime']

        if returnMetric:
            return results[self.evalMetricName.lower()]

        if not processQuiet:
            if batchSize==1:
                myPrint('\r  Processed %d Sample. (Time Per Sample: %.2f ms)%30s\n'%(totalSamples, (1000.0*totalTime)/(maxSamples-1),' '))
            else:
                myPrint('\r  Processed %d Sample. (Time: %.2f Sec.)%30s\n'%(totalSamples, time.time()-t0,' '))

        if not quiet:
            print('MSE:  %f'%(aMse))
            print('RMSE: %f'%(aRmse))
            print('MAE:  %f'%(aMae))
            if aPsnr is not None: print('PSNR: %f'%(aPsnr))
        
        return results

    # ******************************************************************************************************************
    def getMetricVal(self, predicted, actual):
        r"""
        This function calculates and returns the evaluation metric specified by "evalMetricName" property of this dataset class. It is called by the :py:meth:`evaluateModel` function when the "returnMetric" argument is set to True.
        
        Parameters
        ----------
        predicted : array
            The predicted values of the output for the evaluation samples. This is a 1-D arrays of labels for Classification problems or an array of output tensors for Regression problems.
            
        actual : array
            The actual values of the output for the evaluation samples.

        Returns
        -------
        float
            The calculated value of the metric value for this dataset.
        """
        if self.numClasses==0:
            # Regression:
            # Supported Metrics: "MSE", "RMSE, ""MAE", "PSNR"
            # If we are here, this means this is a regression of scaler values. Multi-Dimensional
            # Regression is handled in "evalMultiDimRegression" function above.
            numSamples = len(actual)
            if self.evalMetricName == 'MSE':    return np.square(predicted-actual).mean()
            if self.evalMetricName == 'RMSE':   return np.sqrt( np.square(predicted-actual).mean() )
            if self.evalMetricName == 'MAE':    return np.abs(predicted-actual).mean()
            if self.evalMetricName == 'PSNR':
                assert self.psnrMax is not None
                rmseClipped = np.clip( np.sqrt( np.square(predicted-actual).mean()), 0.000001, None)
                return 20.*np.log10(self.psnrMax/rmseClipped)

        else:
            # Classification:
            if self.evalMetricName == 'Error':
                errorRate = np.sum(actual != predicted)*100.0/float(self.numSamples)
                return errorRate
        
        raise ValueError("Unsupported metric name '%s'!"%(self.evalMetricName))

    # ******************************************************************************************************************
    def evaluate(self, predicted, actual, topK=0, confMat=False, expAcc=None, quiet=False):
        r"""
        Returns information about evaluation results based on the "predicted" and "actual" values. It calls the "evaluateClassification" function for classification problems and "evaluateRegression" function for regression problems.
        
        Parameters
        ----------
        predicted : array
            The predicted values of the output for the evaluation samples. For classification cases, if topK is not zero, this is a list of arrays each containing the K class indexes with highest probabilities in ascending order (The last one is the best) If topK is zero, then this is a 1-D arrays of predicted labels. For regression cases, this is a list of predicted tensors.
            
        actual : array
            The actual values of the output for the evaluation samples.
            
        topK : int
            For classification cases, this indicates whether a "top-K" accuracy value should also be calculated. For example for ImageNet dataset classification, usually the top-5 accuracy value is used (topK=5) besides the top-1. If it is zero (default), the top-K error is not calculated. This is ignored for regression cases.

        confMat : Boolean
            For classification cases, this indicates whether the confusion matrix should be calculated. If the number of classes is more than 10, this argument is ignored and confusion matrix is not calculated. This is ignored for regression cases.

        expAcc : Boolean or None
            Ignored for regression cases. For classification cases:
                 
            * If this is a True, the expected accuracy and kappa values are also calculated. When the number of classes and/or number of evaluation samples is large, calculating expected accuracy can take a long time.
            * If this is False, the expected accuracy and kappa are not calculated.
            * If this is None (the default), then the expected accuracy and kappa are calculated only if number of classes does not exceed 10.
                                        
            **Note**: If confMat is True, then expAcc is automatically set to True.
                    
        quiet : Boolean
            If False, it prints the test results. The printed information includes the Confusion matrix and accuracy information for Classification problems and MSE, RMS, MAE, and PSNR for Regression problems.
                
        Returns
        -------
        dict
            A dictionary of the results information. See "evaluateClassification" and "evaluateRegression" functions for more information.
        """
        if self.numClasses>0:   return self.evaluateClassification(predicted, actual, topK, confMat, expAcc, quiet)
        return self.evaluateRegression(predicted, actual, quiet)
        
    # ******************************************************************************************************************
    def evaluateRegression(self, predicted, actual, quiet=False):
        r"""
        Returns information about test results for Regression problems based on the "predicted" and "actual" values.
        
        This function is used only when the regression output of the model is a scaler value. For multi-dimensional regression problems the "evalMultiDimRegression" function is used.
        
        Note
        ----
        This function is not usually called directly. You should use the "evaluate" function which calls this function internally if this is a regression dataset.
        
        Parameters
        ----------
        predicted : array
            The predicted values of the output as an arrays of output tensors.
            
        actual : array
            The actual values of the output for the test samples.
            
        quiet : Boolean
            If False, it prints the test results. The printed information includes MSE, RMS, MAE, and PSNR.
                
        Returns
        -------
        dict
            A dictionary of the results information. Here is a list of items in the results dictionary:
            
            * **mse**: Mean Square Error (MSE)
            * **rmse**: Root Mean Square Error (RMSE)
            * **mae**: Mean Absolute Error (MAE)
            * **psnr**: Peak Signal to Noise Ratio (PSNR)
            * **csvItems**: A list of evaluation metrics that will be included in the CSV file when performing a parameter search.
        """
        # Important Note:
        # For scaler regression problems we calculate MSE, RMSE, MAE, and PSNR globally
        numSamples = len(actual)
        mse = np.square(predicted-actual).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(predicted-actual).mean()
        psnr = None if self.psnrMax is None else (20.*np.log10(self.psnrMax/np.clip(rmse, 0.000001, None)))

        results = {
            'mse':          mse,
            'rmse':         rmse,
            'mae':          mae,
            'csvItems':     ['mse','rmse','mae'],
        }
        
        if psnr is not None:
            results['psnr'] = psnr
            results['csvItems'] += ['psnr']
            
        if not quiet:
            print('MSE:  %f'%(mse))
            print('RMSE: %f'%(rmse))
            print('MAE:  %f'%(mae))
            if psnr is not None:  print('PSNR: %f'%(psnr))
        
        return results

    # ******************************************************************************************************************
    def evaluateClassification(self, predicted, actual, topK=0, confMat=False, expAcc=None, quiet=False):
        r"""
        Returns classification metrics and draws a confusion matrix for the results based on the information in the actual and predicted labels.
        
        Parameters
        ----------
        predicted : array
            The predicted values of the output for the evaluation samples. If topK is not zero, this is a list of arrays each containing the K class indexes with highest probabilities in ascending order (the last one is the best). If topK is zero, then this is a 1-D arrays of predicted labels.
            
        actual : array
            The actual values of the output for the test samples.
        
        topK : int
            This indicates whether a "top-K" accuracy value should also be calculated. For example for ImageNet dataset classification, usually the top-5 accuracy value is used (topK=5) besides the top-1. If it is zero (default), the top-K error is not calculated.

        confMat : Boolean
            This indicates whether the confusion matrix should be calculated.
            If the number of classes is more than 10, this argument is ignored
            and confusion matrix is not calculated.

        expAcc : Boolean or None
                 
            * If this is a True, the expected accuracy and kappa values are also calculated. When the number of classes and/or number of evaluation samples is large, calculating expected accuracy can take a long time.
            * If this is False, the expected accuracy and kappa are not calculated.
            * If this is None (the default), then the expected accuracy and kappa are calculated only if number of classes does not exceed 10.
                                        
            **Note**: If confMat is True, then expAcc is automatically set to True.
            
        quiet : Boolean
            If False, it prints the test results. The printed information includes the accuracy information and the Confusion matrix if number of classes is less than 10.
                        
        Returns
        -------
        dict
            A dictionary of the results information. Here is a list of items in the results dictionary:
            
            * **accuracy**: The evaluated accuracy. A float number between 0 and 1. AKA "Observed Accuracy", it is defined as:
              
                .. math:: Observed Accuracy = \frac{TN+TP}{N}
            * **errorRate**: The evaluated error rate (=1-accuracy). A float number between 0 and 1.
            * **expectedAccuracy**: Expected Accuracy of the evaluated. It is defined as:
            
            .. math:: ExpectedAccuracy = \frac{(TN+FN)(TN+FP)+(FP+TP)(FN+TP)}{N^2}
            
            * **kappa**: kappa. It is defined as:
            
                .. math:: \kappa = \frac{Observed Accuracy - Expected Accuracy}{1-Expected Accuracy}
            * **truePositives**: An array containing True Positive values (TP) for each class.
            * **precisions**: An array containing Precision values for each class. The Precision for each class it is defined as:
            
                .. math:: Precision = \frac{TP}{TP+FP}
            * **recalls**: An array containing Recall values for each class. The Recall for each class it is defined as:
            
                .. math:: Recall = \frac{TP}{TP+FN}
            * **fMeasures**: An array containing fMeasures values for each class. The fMeasures for each class it is defined as:
            
                .. math:: f = \frac{2*Precision*Recall}{Precision+Recall}
            * **confusionAP**: A 2-D nxn array where n is the number of classes. confusionAP[a][p] is the number of samples that were predicted in class 'p' and are actually in class 'a'.
                        
            * **top<N>Accuracy**: Included only if topK is not zero. The <N> in the key name is replaced with the actual value of topK.
            
            * **csvItems**: A list of evaluation metrics that will be included in the CSV file when performing a parameter search.

            In the above equations:
            
            ====  =============
            TN    True Negatives
            TP    True Positives
            FN    False Negatives
            FP    False Positives
            ====  =============
        """
        CONFUSION_MATRIX_MIN_CLASS = 10
        if expAcc is None:  expAcc = (self.numClasses<=CONFUSION_MATRIX_MIN_CLASS)

        if self.numClasses>CONFUSION_MATRIX_MIN_CLASS:  confMat = False
        if confMat: expAcc = True   # expected accuracy is needed for confusion matrix
        
        if topK>0:
            if type(predicted)==list:       predicted=np.array(predicted)
            if type(actual)==list:          actual=np.array(actual)

            assert (self.numClasses>topK), "Cannot calculate Top-%d accuracy! (Number of classes: %d)"%(topK, self.numClasses)
            topKAccuracy = np.sum(np.prod(actual.reshape((-1,1))-predicted,-1)==0)/float(actual.shape[0])
            predicted = predicted[:,-1]

        numSamples = len(actual)
        sumActuals = [0]*self.numClasses
        sumPreds = [0]*self.numClasses
        
        truePos = None
        precisions = None
        recalls = None
        fMeasures = None
        confusionAP = None
        expAccuracy = None
        kappa = None
        
        if expAcc:
            truePos = [0]*self.numClasses
            precisions = [0.0]*self.numClasses
            recalls = [0.0]*self.numClasses
            fMeasures = [0.0]*self.numClasses
            confusionAP = [ [0]*self.numClasses for i in range(self.numClasses)]
            
            for r in range(self.numClasses):
                actualR = [i for i in range(len(actual)) if actual[i]==r]
                if truePos is not None:
                    truePos[r] = len([i for i in actualR if predicted[i]==r])
                
                for c in range(self.numClasses):
                    actualPredRC = len([i for i in actualR if predicted[i]==c])
                    sumPreds[c] += actualPredRC
                    sumActuals[r] += actualPredRC
                    if confusionAP is not None: confusionAP[r][c] = actualPredRC

            if recalls is not None:
                for c in range(self.numClasses):
                    if sumActuals[c]>0: recalls[c] = float(truePos[c])/float(sumActuals[c])
                    if sumPreds[c]>0:   precisions[c] = float(truePos[c])/float(sumPreds[c])
                    if (recalls[c]+precisions[c])>0:
                        fMeasures[c] = (2.0*recalls[c]*precisions[c])/(recalls[c]+precisions[c])

            expAccuracy = float(sum([sumActuals[i]*sumPreds[i] for i in range(self.numClasses)]))/float(numSamples*numSamples)

        accuracy = 1.0 - np.sum(np.int32(actual) != np.int32(predicted))/float(numSamples)
        if (expAccuracy is not None) and (expAccuracy!=1.0):
            kappa = (accuracy-expAccuracy)/(1.0-expAccuracy)

        results = {
            'accuracy': accuracy,
            'errorRate': 1-accuracy,
            'csvItems': ['accuracy','errorRate'],
        }
        
        if expAcc:
            results['expectedAccuracy'] = expAccuracy
            results['kappa'] = kappa
            results[ 'csvItems' ] += ['expectedAccuracy','kappa']
            results['truePositives'] = truePos
            results['precisions'] = precisions
            results['recalls'] = recalls
            results['fMeasures'] = fMeasures
            results['confusionAP'] = confusionAP    # 2-D array: confusionAP[a][p] is the number of samples that were
                                                    # predicted in class 'p' and are actually in class 'a'
        if topK>0:
            results[ 'top%dAccuracy'%topK ] = topKAccuracy
            results[ 'csvItems' ].insert(3, 'top%dAccuracy'%topK)

        if not quiet:
            if confMat:
                print('\nConfusion Matrix:\n             +' + (9*self.numClasses-1+16)*'-' + '+\n' +
                      '             | Predicted' + (9*self.numClasses - 11 + 16)*' ' + '|')
                sepLine = '+'
                header =  '|'
                for c in range(self.numClasses):
                    sepLine += '--------+'
                    header += (' %-6s |' % str(c))
                sepLineEx = sepLine + '---------------+'
                header +=             ' TP     Recall |'
                print('             ' + sepLineEx +
                      '\n             ' + header +
                      '\n+---+--------' + sepLineEx)
                actualCol = 'Actual' + self.numClasses*' '
                for r in range(self.numClasses):
                    row = ('| %s | %-6s |' % (actualCol[r], str(r)))
                    for c in range(self.numClasses):
                        row += (' %-6d |' % confusionAP[r][c])
                    row += (' %-6d %5.2f%% |' % (truePos[r], recalls[r]*100.0))
                    print(row)
                print( '+------------' + sepLineEx )

                row = ('| Precision  |')
                for c in range(self.numClasses):   row += (' %5.2f%% |' % (precisions[c]*100.0))
                print(row + '\n+------------' + sepLine)

                row = ('| F-measure  |')
                for c in range(self.numClasses):   row += ('  %-6.2f|' % (fMeasures[c]))
                print(row + '\n+------------' + sepLine)

            print('Observed Accuracy: %f'%(accuracy))
            if expAccuracy is not None: print('Expected Accuracy: %f'%(expAccuracy))
            if topK>0:                  print('Top-%d Accuracy:   %f'%(topK, topKAccuracy))
            if kappa is not None:
                kappaStrs = ['Poor', 'Fair', 'Moderate', 'Good', 'Excellent']
                print('Kappa: %f (%s)'%(kappa, kappaStrs[ max(int((kappa-.01)/.2),0) ]))
        return results

    # ******************************************************************************************************************
    @classmethod
    def downloadAndExtractZipFile(cls, zipFileUrl, destFolder):
        # Example fir zipFileUrl: "http://images.cocodataset.org/zips/val2017.zip"
        # Example for destFolder: "/data/mscoco/"
        destFileName = zipFileUrl[zipFileUrl.rfind('/')+1:] # Example: "val2017.zip"
        dotExt = zipFileUrl[zipFileUrl.rfind('.'):]         # Example: ".zip"
        destFilePath = destFolder + destFileName            # Example:  /data/mscoco/val2017.zip
        destExtractionFolder = destFilePath[:-len(dotExt)]  # Example: "/data/mscoco/val2017"
        
        if os.path.exists(destExtractionFolder):
            return True

        if not os.path.exists(destFilePath):
            # Example: /data/mscoco/val2017.zip does not exist
            try:
                print('Downloading from "%s" ...'%(zipFileUrl))
                urlreq.urlretrieve(zipFileUrl, destFilePath)
            except:
                # Failed to download from the given folder. We can try again below from other locations
                print('  Failed!')
                if os.path.exists(destFilePath): os.remove(destFilePath)
                return False

        try:
            print('Extracting "%s" ...'%(destFilePath))
            with zipfile.ZipFile(destFilePath, 'r') as zipFile:
                zipFile.extractall(destFolder)
            print('Deleting "%s" ...'%(destFilePath))
            os.remove(destFilePath)
            return True
            
        except:
            # Failed to extract! Maybe zip file was corrupted. Deleting the zip file.
            print('  Failed!')
            print('Deleting "%s" ...'%(destFilePath))
            os.remove(destFilePath)
                
        return False

    # ******************************************************************************************************************
    @classmethod
    def download(cls, folderName, files, destDataFolder=None):
        r"""
        This class method can be called to download dataset files from their original source or a
        Fireball online repository. This method is usually called internally by one of the
        derived classes.
        
        Parameters
        ----------
        folderName: str
            A string containing the name of the dataset folder. This is used both for Fireball
            repository and the destination folder name on local machine.

        files: list of str
            A list of strings. Each item in the list can be a file name or a URL.
            
                * URL: In this case the URL is tried first (which is usually the original location of the dataset files). If for any reason the file cannot be downloaded then the file name is extracted from the URL and it is downloaded from the Fireball repository.
                * Name: In this case the file is downloaded directly from the Fireball repository.
            
            If the downloaded file is a zip file, it is extracted to the dataset directory.
            
        destDataFolder: str
            The folder where dataset folders and files are saved. If this is not provided, then
            a folder named "data" is created in the home directory of the current user and the
            dataset folders and files are created there.
        """
        if destDataFolder is None: destDataFolder = os.path.expanduser("~")+'/data'

        if destDataFolder[-1] != '/':   destDataFolder+='/' # Example: /data/ or /home/shahab/data/
        if folderName[-1] != '/':       folderName+='/'     # Example: mscoco/
        destFolder = destDataFolder + folderName            # Example: /data/mscoco/
        if not os.path.exists(destFolder):
            print('Creating folder "%s" ...'%(destFolder))
            os.makedirs(destFolder)
        elif os.path.isfile(destFolder):
            raise ValueError("'%s' must be a directory but it is a file!"%(destFolder))

        alreadyDownloaded = set()
        for file in files:
            # Examples for file: 'http://images.cocodataset.org/zips/val2017.zip' or 'train2014.zip'
            if file[:7].lower()!='http://':                     continue
            destFileName = file[file.rfind('/')+1:]             # Example: val2017.zip
            destFilePath = destFolder + destFileName            # Example:  /data/mscoco/val2017.zip
            dotExt = file[file.rfind('.'):]                     # Example:  .zip
            isZip = dotExt in ['.zip', '.gz']
            if isZip:
                if cls.downloadAndExtractZipFile(file, destFolder): alreadyDownloaded.add(destFileName)
                continue
            
            elif not os.path.exists(destFilePath):
                # Not a zip file
                # Example: /data/ImageNet/TrainDataset.csv does not exist
                try:
                    print('Downloading from "%s" ...'%(file))
                    urlreq.urlretrieve(file, destFilePath)
                    alreadyDownloaded.add(destFileName)

                except:
                    # Failed to download from the given folder. We can try again below from other locations
                    print('  Failed!')
                    if os.path.exists(destFilePath): os.remove(destFilePath)
                    continue
            else:
                # Not a zip file and already exists
                alreadyDownloaded.add(destFileName)

        # Some of the files have already been downloaded and are in 'alreadyDownloaded' set.
        locInfoUrl = "https://interdigitalinc.github.io/Fireball/LocInfo.yml"
        locInfo = yaml.safe_load(urlreq.urlopen(locInfoUrl).read())
        for location in locInfo['dsetLocations']:
            if location[-1] != '/': location+='/'
            for file in files:
                destFileName = file[file.rfind('/')+1:] if file[:7].lower()=='http://' else file  # Example: val2017.zip
                if destFileName in alreadyDownloaded: continue
                
                fileUrl = location + folderName + destFileName
                destFilePath = destFolder + destFileName            # Example:  /data/mscoco/val2017.zip
                dotExt = file[file.rfind('.'):]                     # Example:  .zip
                isZip = dotExt in ['.zip', '.gz']

                if isZip:
                    if cls.downloadAndExtractZipFile(fileUrl, destFolder): alreadyDownloaded.add(destFileName)
                    continue
                
                elif not os.path.exists(destFilePath):
                    # Not a zip file
                    # Example: /data/ImageNet/TrainDataset.csv does not exist
                    try:
                        print('Downloading from "%s" ...'%(fileUrl))
                        urlreq.urlretrieve(fileUrl, destFilePath)
                        alreadyDownloaded.add(destFileName)

                    except:
                        # Failed to download from the given folder. We can try again below from other locations
                        print('  Failed!')
                        if os.path.exists(destFilePath): os.remove(destFilePath)
                        continue
                else:
                    # Not a zip file and already exists
                    alreadyDownloaded.add(destFileName)
