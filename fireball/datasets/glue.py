# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of `GLUE <https://gluebenchmark.com>`_ dataset class for different NLP tasks. Use the ``GlueDSetUnitTest.py`` file in the ``UnitTest/Datasets`` folder to run the Unit Test of this implementation.

This implementation assumes that the following folders exist in the 'dataPath' directory for each one of supported GLUE tasks. For more information about GLUE tasks please refer to: https://gluebenchmark.com/tasks

    * ``CoLA``: The Corpus of Linguistic Acceptability
    * ``SST-2``: The Stanford Sentiment Treebank
    * ``MRPC``: Microsoft Research Paraphrase Corpus
    * ``STS-B``: Semantic Textual Similarity Benchmark
    * ``QQP``: Quora Question Pairs
    * ``MNLI``: MultiNLI (Matched and Mismatched)
    * ``QNLI``: Question NLI
    * ``RTE``: Recognizing Textual Entailment
    * ``WNLI``: Winograd NLI
    * ``SNLI``: Stanford NLI Corpus (Not officially part of GLUE)
    * ``AX``: Auxiliary Task (GLUE Diagnostic Dataset)
   
Note
----
    * If there is a "vocab.txt" file in the 'dataPath' directory and no tokenizer is specified, this file is used to create a tokenizer.
    * While reading dataset files some of the samples may be dropped if the sequence length exceeds the 'maxSeqLen'

**Tasks**

CoLA
    The Corpus of Linguistic Acceptability (Metric: Matthew's Corr)

    +------------------+------------------+------------------+------------------+
    | Class            | Training Samples | Dev Samples      | Test Samples     |
    +==================+==================+==================+==================+
    |   0 Unacceptable | 2528   (29.56% ) | 322    (30.87%)  | Unknown          |
    |                  |                  |                  |                  |
    |   1 Acceptable   | 6023   (70.44%)  | 721    (69.13%)  | Unknown          |
    +------------------+------------------+------------------+------------------+
    | Total            | 8551   (80.24% ) | 1043    (9.79%)  | 1063    (9.97%)  |
    +------------------+------------------+------------------+------------------+
    | Max Seq. len     | 47               | 35               | 38               |
    |                  |                  |                  |                  |
    | Samples Dropped  | 0                | 0                | 0                |
    +------------------+------------------+------------------+------------------+

SST-2
    The Stanford Sentiment Treebank (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 Negative       | 29780   (44.22%) | 428    (49.08%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Positive       | 37569   (55.78%) | 444    (50.92%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 67349   (96.16%) | 872     (1.24%)  | 1821    (2.60%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 66               | 55               | 64               |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+

MRPC
    Microsoft Research Paraphrase Corpus (Metric: F1/Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 Irrelevant     | 1194   (32.55%)  | 129    (31.62%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Equivalent     | 2474   (67.45%)  | 279    (68.38%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 3668   (63.23%)  | 408     (7.03%)  | 1725   (29.74%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 103              | 86               | 104              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+

STS-B
    Semantic Textual Similarity Benchmark (Metric: Pearson-Spearman Corr)

    +--------------------+------------------+------------------+------------------+
    | Parameter          | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    | Total Count        | 5749             | 1500             | 1379             |
    |                    |                  |                  |                  |
    | Max Seq. len       | 125              | 87               | 81               |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+

QQP
    Quora Question Pairs (Metric: F1/Accuracy)
    
    **NOTE**: This dataset contains some invalid records in train and dev files which are ignored.

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 Different      | 229471  (63.07%) | 25545   (63.18%) | Unknown          |
    |                    |                  |                  |                  |
    |   1 Duplicate      | 134378  (36.93%) | 14885   (36.82%) | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 363849  (45.75%) | 40430   (5.08%)  | 390965  (49.16%) |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 330              | 199              | 319              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+

MNLI-m
    MultiNLI (Matched) (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 Contradiction  | 130903 (33.33%)  | 3213   (32.74%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Entailment     | 130899 (33.33%)  | 3479   (35.45%)  | Unknown          |
    |                    |                  |                  |                  |
    |   2 Neutral        | 130900 (33.33%)  | 3123   (31.82%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 392702 (95.24%)  | 9815    (2.38%)  | 9796    (2.38%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 444              | 237              | 249              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+

MNLI-mm
    MultiNLI (Mismatched) (Metric: Accuracy)
    
    **NOTE**: This task is only available for 'dev' and 'test' datasets. It should be used with the model trained for the ``MNLI-m`` task.

    +--------------------+------------------+------------------+
    | Class              | Dev Samples      | Test Samples     |
    +====================+==================+==================+
    |   0 Contradiction  | 3240   (32.95%)  | Unknown          |
    |                    |                  |                  |
    |   1 Entailment     | 3463   (35.22%)  | Unknown          |
    |                    |                  |                  |
    |   2 Neutral        | 3129   (31.82%)  | Unknown          |
    +--------------------+------------------+------------------+
    | Total              | 9832   (2.38%)   | 9847    (2.39%)  |
    +--------------------+------------------+------------------+
    | Max Seq. len       | 211              | 262              |
    |                    |                  |                  |
    | Samples Dropped    | 0                | 0                |
    +--------------------+------------------+------------------+
    
QNLI
    Question NLI (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 NotEntailment  | 52366  (50.00%)  | 2761   (50.54%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Entailment     | 52372  (50.00%)  | 2702   (49.46%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 104738 (90.55%)  | 5463    (4.72%)  | 5463    (4.72%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 307              | 250              | 294              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 5                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+
    
    **Note**: Using maxSeqLen=384, 5 training samples are dropped because they result in longer sequences.

RTE
    Recognizing Textual Entailment (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 NotEntailment  | 1241   (49.84%)  | 131    (47.29%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Entailment     | 1249   (50.16%)  | 146    (52.71%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 2490   (43.18%)  | 277     (4.80%)  | 3000   (52.02%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 289              | 253              | 252              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+
    
WNLI
    Winograd NLI (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 NotEntailment  | 323    (50.87%)  | 40     (56.34%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Entailment     | 312    (49.13%)  | 31     (43.66%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 635    (74.53%)  | 71      (8.33%)  | 146    (17.14%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 108              | 105              | 100              |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+
    
SNLI
    Stanford NLI Corpus (Not officially part of GLUE) (Metric: Accuracy)

    +--------------------+------------------+------------------+------------------+
    | Class              | Training Samples | Dev Samples      | Test Samples     |
    +====================+==================+==================+==================+
    |   0 Contradiction  | 183187 (33.35%)  | 3278   (33.31%)  | Unknown          |
    |                    |                  |                  |                  |
    |   1 Entailment     | 183416 (33.39%)  | 3329   (33.82%)  | Unknown          |
    |                    |                  |                  |                  |
    |   2 Neutral        | 182764 (33.27%)  | 3235   (32.87%)  | Unknown          |
    +--------------------+------------------+------------------+------------------+
    | Total              | 549367 (96.54%)  | 9842    (1.73%)  | 9824    (1.73%)  |
    +--------------------+------------------+------------------+------------------+
    | Max Seq. len       | 71               | 59               | 36               |
    |                    |                  |                  |                  |
    | Samples Dropped    | 0                | 0                | 0                |
    +--------------------+------------------+------------------+------------------+
    
AX
    Auxiliary Task (GLUE Diagnostic Dataset)
    
    This task is only available for 'test' dataset. It is used when submitting results to GLUE website. The labels should be predicted using the model trained by ``MNLI-M`` task. The labels for the samples in this dataset are unknown. Here are some statistics:
    
        * Total Samples:   1104
        * Max Seq. len:    121
        * Samples Dropped: 0
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 08/21/2020    Shahab Hamidi-Rad       Created the file.
# 08/24/2020    Shahab Hamidi-Rad       Finished implementing all GLUE tasks (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI
#                                       RTE, WNLI, and SNLI)
# 02/09/2021    Shahab Hamidi-Rad       Added support for the AX task. Also added support for GLUE website submission.
# **********************************************************************************************************************
import json
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint

# **********************************************************************************************************************
class GlueDSet(BaseDSet):
    r"""
    This class implements the GLUE group of datasets.
    """
    classNames = None
    numClasses = 0
    
    # ******************************************************************************************************************
    def __init__(self, taskName, dsName='Train', dataPath=None, batchSize=8, tokenizer=None, numWorkers=0):
        r"""
        Constructs a `GlueDSet` instance. This can be called directly or via :py:meth:`makeDatasets` class method.
        
        Parameters
        ----------
        taskName : str
            One of the GLUE task names. Currently the following tasks are supported:
                * ``"CoLA"``: The Corpus of Linguistic Acceptability
                * ``"SST-2"``: The Stanford Sentiment Treebank
                * ``"MRPC"``: Microsoft Research Paraphrase Corpus
                * ``"STS-B"``: Semantic Textual Similarity Benchmark
                * ``"QQP"``: Quora Question Pairs
                * ``"MNLI-M"``: MultiNLI Matched
                * ``"MNLI-MM"``: MultiNLI Mismatched
                * ``"QNLI"``: Question NLI
                * ``"RTE"``: Recognizing Textual Entailment
                * ``"WNLI"``: Winograd NLI
                * ``"SNLI"``: Stanford NLI Corpus
                * ``"AX"``: Auxiliary Task (GLUE Diagnostic Dataset)

        dsName : str
            The name of the dataset. It can be one "Train", "Dev", or "Test".

        dataPath : str
            The path to the directory where the dataset files are located.
            
        batchSize : int
            The default batch size used in the "batches" method.
            
        tokenizer : Tokenizer object
            The tokenizer used to tokenize the text info in the dataset files.
            
        numWorkers : int
            The number of worker threads used to load the samples.
        """
        self.taskName = taskName
        taskLo = self.taskName.lower()
        if taskLo == 'cola':                                GlueDSet.classNames = ['Unacceptable', 'Acceptable']
        elif taskLo == 'sst-2':                             GlueDSet.classNames = ['Negative', 'Positive']
        elif taskLo == 'mrpc':                              GlueDSet.classNames = ['Irrelevant', 'Equivalent']
        elif taskLo == 'sts-b':                             GlueDSet.classNames = None
        elif taskLo == 'qqp':                               GlueDSet.classNames = ['Different', 'Duplicate']
        elif taskLo in ['mnli-m', 'mnli-mm', 'snli', 'ax']: GlueDSet.classNames = ['Contradiction', 'Entailment', 'Neutral']
        elif taskLo in ['qnli', 'rte', 'wnli']:             GlueDSet.classNames = ['NotEntailment', 'Entailment']
        else:                                               raise ValueError("Unsupported Task \"%s\"!"%(self.taskName))
        GlueDSet.numClasses = 0 if GlueDSet.classNames is None else len(GlueDSet.classNames)
        
        self.maxSeqLen = {"cola": 64, "qqp":384, "mnli-m":448, "mnli-mm":448, "qnli":384, "rte":384}.get(taskLo, 128)

        if dataPath is None:
            dataPath = '/data/GLUE/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
        assert tokenizer is not None, "'tokenizer' cannot be None!"
       
        self.tokenizer = tokenizer
        super().__init__(dsName, dataPath, None, None, batchSize, numWorkers)
        self.sampleShape = (self.maxSeqLen,)        # Length of question+context (padded to "maxSeqLen")

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
        repStr += '    Task ........................................... %s\n'%(self.taskName)
        return repStr

    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This function is called by the constructor of :py:class:`BaseDSet` to load the samples and labels of this dataset based on `taskName`, `dsName`, and `dataPath`.
        """
  
        self.samples = []       # Each sample is: (<id>, <tokIds>, <tokTypes>, <noPadLen>)
        self.sampleIds = []     # Converts a sample index to original sampleId
        self.labels = []
        maxTokIdsLen = [0,0,0]  # seqLen (text1+text2+3), text1, text2
        numSamplesDropped = 0   # Number of samples dropped because the length exceeded self.maxSeqLen
        
        taskLo = self.taskName.lower()
        folderNames = { 'cola': "CoLA", 'mnli-m':"MNLI", 'mnli-mm': "MNLI", 'ax': "diagnostic" }
        dsFileSuffix = { 'mnli-m': "_matched", 'mnli-mm': "_mismatched" }
        fileName = "%s%s/%s%s.tsv"%(self.dataPath, folderNames.get(taskLo, self.taskName.upper()),
                                    'diagnostic' if taskLo == 'ax' else self.dsName.lower(),
                                    '' if self.dsName.lower()=='train' else dsFileSuffix.get(taskLo, ''))
        
        # Field indexes for text1, text2, and label for different datasets:
        t1, t2, l = None, None, None
        if taskLo == 'cola':                            t1,l = 3, 1
        elif taskLo == 'sst-2':                         t1,l = 0, 1
        elif taskLo == 'mrpc':                          t1,t2,l = 3, 4, 0
        elif taskLo == 'sts-b':                         t1,t2,l = 7, 8, 9
        elif taskLo in ['qnli', 'rte', 'wnli']:         t1,t2,l = 1, 2, 3
        elif taskLo == 'qqp':                           t1,t2,l = 3, 4, 5
        elif taskLo in ['mnli-m', 'mnli-mm', 'snli']:   t1,t2,l = 8, 9, -1
        elif taskLo == 'ax':                            t1,t2,l = 1, 2, 0
        
        if taskLo == 'ax':
            assert self.dsName.lower() == 'test', "The 'AX' task (Diagnostic Dataset) is only available as 'test' dataset!"

        if taskLo == 'mnli-mm':
            assert self.dsName.lower() != 'train', "The 'MNLI-MM' task is only available as 'dev' or 'test' dataset!"

        isTest = (self.dsName.lower() == 'test')
        if isTest:
            if taskLo in ['cola', 'sst-2']: t1 = 1
            elif taskLo == 'qqp':           t1,t2 = 1,2
            l=0
        
        def getLabel(labelStr):
            if isTest: return 0
            if taskLo in ['mnli-m', 'mnli-mm', 'snli', 'ax']: return {'contradiction':0, 'entailment':1, 'neutral':2}[labelStr]
            if taskLo in ['qnli', 'rte']:                     return {'not_entailment':0, 'entailment':1}[labelStr]
            if taskLo == 'sts-b':                             return float(labelStr)
            return int(labelStr)
        
        headerLines = 1
        if (taskLo == 'cola') and (not isTest): headerLines = 0

        with open(fileName, "r") as tsvFile: lines = tsvFile.readlines()
        records = [line.split('\t') for line in lines[headerLines:]]

        recLen = len(records[0])
        assert recLen>t1, "%d - %d"%(recLen, t1)
        if t2 is not None:  assert recLen>t2, "%d - %d"%(recLen, t2)
        if not isTest:      assert recLen>l, "%d - %d"%(recLen, l)
        for i,record in enumerate(records):
            if len(record)!=recLen and taskLo!='snli':
                # SNLI has records with different lengths. The last item is always label (l=-1) and t1,t2 are 8 and 9
                # QQP has some invalid records, which will cause the following warning message.
                # Other than these two cases, the following warning should be seen.
                myPrint('Warning: Skipping Invalid record in "%s" line %d!'%(fileName, i+headerLines+1), color='yellow')
                continue
            
            text1Tokens, _ = self.tokenizer.encode(record[t1])
            text1Len = len(text1Tokens)
            seqLen = text1Len + 2

            text2Tokens = None
            text2Len = 0
            if t2 is not None:
                text2Tokens, _ = self.tokenizer.encode(record[t2])
                text2Len = len(text2Tokens)
                seqLen += text2Len + 1
                
            if seqLen>self.maxSeqLen:
                numSamplesDropped += 1
                continue
             
            self.sampleIds += [ "%s-%d"%(self.dsName.lower(),i) ]
            sample = self.tokenizer.packTokenIds(text1Tokens, text2Tokens, self.maxSeqLen)
            self.samples += [ sample ]
            
            self.labels += [ getLabel( record[l].strip() ) ]
            
            if text1Len > maxTokIdsLen[1]:      maxTokIdsLen[1] = text1Len
            if text2Len > maxTokIdsLen[2]:      maxTokIdsLen[2] = text2Len
            if seqLen > maxTokIdsLen[0]:        maxTokIdsLen[0] = seqLen
        
        self.stats = {"maxText1Len":          maxTokIdsLen[1],
                      "maxText2Len":          maxTokIdsLen[2],
                      "maxSeqLen":            maxTokIdsLen[0],
                      "numSamplesDropped":    numSamplesDropped }
        
        if self.taskName.lower() == 'sts-b':    self.labels = np.float32(self.labels)
        else:                                   self.labels = np.int32(self.labels)

    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, taskName, dsNames='Train,Dev,Test', batchSize=8, dataPath=None, tokenizer=None, numWorkers=0):
        r"""
        This class method creates several datasets in one-shot as specified by `dsNames` parameter.
        
        Parameters
        ----------
        taskName : str
            One of the GLUE task names. Please refer to the documentation for the :py:meth:`__init__` method above for more details about supported tasks.
                
        dsNames : str
            A combination of the following:
            
                * ``Train``: Create the training dataset.
                * ``Dev``:  Create the dev dataset.
                * ``Test``:  Create the test dataset. (Labels unknown)

        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        tokenizer : Tokenizer object
            The tokenizer used by all created datasets. If this is None, and there is a ``vocab.txt`` file in the ``dataPath``, this method tries to create a tokenizer using ``vocab.txt`` as its vocabulary.
            
        numWorkers : int
            The number of worker threads used to load the samples.

        Returns
        -------
        Up to 3 :py:class:`GlueDSet` objects
            Depending on the number of items specified in the `dsNames`, it returns one to three GlueDSet objects. The returned values have the same order as they appear in the `dsNames` parameter.
        """
        if dataPath is None:
            dataPath = '/data/GLUE/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        if tokenizer is None:
            if os.path.exists(dataPath+'vocab.txt'):
                myPrint('Initializing tokenizer from "%s" ... '%(dataPath+'vocab.txt'), False)
                from ..textio import Tokenizer
                tokenizer = Tokenizer(dataPath+'vocab.txt')
                myPrint('Done. (Vocab Size: %d)'%(tokenizer.vocabSize))
            else:
                raise ValueError("A tokenizer is needed for the GLUE dataset to work properly!")

        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            retVals += [ cls(taskName, dsStr, dataPath, batchSize, tokenizer, numWorkers) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals

    # ******************************************************************************************************************
    @classmethod
    def printDsInfo(cls, trainDs=None, devDs=None, testDs=None):
        r"""
        This class method prints information about given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : any object derived from `BaseDSet`, optional
            The training dataset.
            
        devDs : any object derived from `BaseDSet`, optional
            The dev dataset.
        
        testDs : any object derived from `BaseDSet`, optional
            The test dataset.
        """
        
        maxSeqLen = 0
        for ds in [trainDs, devDs, testDs]:
            if ds is None: continue
            dataPath = trainDs.dataPath
            taskName = trainDs.taskName
            numWorkers = trainDs.numWorkers
            maxSeqLen = trainDs.maxSeqLen
            break
            
        if maxSeqLen==0:
            raise ValueError("At least one of 'Train', 'Dev', or 'Test' datasets must be specified!")

        print('%s Dataset Info:'%(cls.__name__))
        print('    Task ........................................... %s'%(taskName))
        print('    Dataset Location ............................... %s'%(dataPath))
        print('    Max Seq. Len ................................... %d'%(maxSeqLen))
        if cls.numClasses>0:
            print('    Number of Classes .............................. %d'%(cls.numClasses))
        if trainDs is not None:
            print('    Number of Training Samples ..................... %d'%(trainDs.numSamples))
        if devDs is not None:
            print('    Number of Dev. Samples ......................... %d'%(devDs.numSamples))
        if testDs is not None:
            print('    Number of Test Samples ......................... %d'%(testDs.numSamples))
        if numWorkers>0:
            print('    Number of Worker Threads ....................... %d'%(numWorkers))

    # ******************************************************************************************************************
    @classmethod
    def printStats(cls, trainDs=None, devDs=None, testDs=None):
        r"""
        This class method prints statistics of the given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : any object derived from `BaseDSet`, optional
            The training dataset.
            
        devDs : any object derived from `BaseDSet`, optional
            The dev dataset.

        testDs : any object derived from `BaseDSet`, optional
            The test dataset. (Unknown Labels)
        """
        maxClassWidth = 14
        if cls.numClasses>0:
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

            if devDs is not None:
                if devDs.labels is None:  return
                devCounts = np.zeros(cls.numClasses)
                for _,label in devDs.batches(1): devCounts[ label[0] ] += 1
                all += devDs.numSamples
                #           1234567890123456 |
                sep +=    '------------------+'
                rowStr += ' Dev Samples      |'

            if testDs is not None:
                all += testDs.numSamples
                #           1234567890123456 |
                sep +=    '------------------+'
                rowStr += ' Test Samples     |'

            print(sep)
            print(rowStr)
            print(sep)

            for c in range(cls.numClasses):
                if str(c) == cls.classNames[c]: rowStr = ('    | %%-%dd |'%(4+maxClassWidth))%(c)    # Mostly for MNIST!
                else:                           rowStr = ('    | %%3d %%-%ds |'%(maxClassWidth))%(c, cls.classNames[c])
                if trainDs is not None:
                    rowStr += ' %-8d %6.2f%% |'%(trainCounts[c], trainCounts[c]*100.0/trainDs.numSamples)
                if devDs is not None:
                    rowStr += ' %-8d %6.2f%% |'%(devCounts[c], devCounts[c]*100.0/devDs.numSamples)
                if testDs is not None:
                    rowStr += ' Unknown          |'

                print( rowStr )
           
            #             | 123 1234567890123 |
            rowStr = '    | Total' + ' '*maxClassWidth + '|'
            if trainDs is not None:
                rowStr += ' %-8d %6.2f%% |' % (trainDs.numSamples, 100.0*trainDs.numSamples/all)
            if devDs is not None:
                rowStr += ' %-8d %6.2f%% |' % (devDs.numSamples, 100.0*devDs.numSamples/all)
            if testDs is not None:
                rowStr += ' %-8d %6.2f%% |' % (testDs.numSamples, 100.0*testDs.numSamples/all)

            print(sep)
            print(rowStr)
            print(sep)
            
        else:
            #             | 123456789012345678 |
            sep =    '    +--------------------+'
            rowStr = '    | Parameter          |'
            
            if trainDs is not None:
                #           1234567890123456 |
                sep +=    '------------------+'
                rowStr += ' Training Samples |'

            if devDs is not None:
                #           1234567890123456 |
                sep +=    '------------------+'
                rowStr += ' Dev Samples      |'

            if testDs is not None:
                #           1234567890123456 |
                sep +=    '------------------+'
                rowStr += ' Test Samples     |'

            print(sep)
            print(rowStr)
            print(sep)

            rowStr = '    | Total Count ' + ' '*(maxClassWidth-7) + '|'
            if trainDs is not None:     rowStr += ' %-16d |'%(trainDs.numSamples)
            if devDs is not None:       rowStr += ' %-16d |'%(devDs.numSamples)
            if testDs is not None:      rowStr += ' %-16d |'%(testDs.numSamples)
            print(rowStr)

        rowStr = '    | Max Seq. len' + ' '*(maxClassWidth-7) + '|'
        if trainDs is not None:     rowStr += ' %-16d |'%(trainDs.stats['maxSeqLen'])
        if devDs is not None:       rowStr += ' %-16d |'%(devDs.stats['maxSeqLen'])
        if testDs is not None:      rowStr += ' %-16d |'%(testDs.stats['maxSeqLen'])
        print(rowStr)
        
        rowStr = '    | Samples Dropped' + ' '*(maxClassWidth-10) + '|'
        if trainDs is not None:     rowStr += ' %-16d |'%(trainDs.stats['numSamplesDropped'])
        if devDs is not None:       rowStr += ' %-16d |'%(devDs.stats['numSamplesDropped'])
        if testDs is not None:       rowStr += ' %-16d |'%(testDs.stats['numSamplesDropped'])
        print(rowStr)
        print(sep)
       
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
        samples : tuple
            This is a 4-tuple::
            
                samples = (batchSampleIdxs, batchTokenIds, batchTokenTypes)
            
            Where:
            
                * **batchSampleIdxs**: 1D list of sample indexes for the samples in this batch.
                * **batchTokenIds**: 2D list of integer tokenId values for 'n' sequences (where 'n' is current batch size). Each sequence (row) contains 'maxSeqLen' tokens including CLS, SEP, and paddings
                * **batchTokenTypes**: 2D list of integers one for each tokenId in "batchTokenIds". '0' is used for first text tokens and '1' for the second text tokens. '0' is also used for the padding tokens.
                
        labels : numpy array of int32 or float32
            For classification tasks this contains the label for each one of batch samples. For regression tasks, this contains the ground-truth value for each one of the batch samples.
        """
        batchComponents = None
        for i in batchIndexes:
            sampleComponents = self.samples[ i ]
            if batchComponents is None:
                batchComponents = [ [] for _ in range(len(sampleComponents)+1) ]
            batchComponents[0] += [ i ] # First component is always sample indexes
            for s, sampleComponent in enumerate(sampleComponents):
                batchComponents[s+1] += [ sampleComponent ]
                            
        return tuple(batchComponents), self.labels[batchIndexes]

    # ******************************************************************************************************************
    def evaluate(self, predicted, actual, topK=0, confMat=False, expAcc=None, quiet=False):
        r"""
        Returns information about evaluation results based on the "predicted" and "actual" values. This is usually called by the :py:meth:`~fireball.datasets.base.BaseDSet.evaluateModel` method which should be called to evaluate a model with this dataset.
        
        Parameters
        ----------
        predicted : array
            The predicted values of the output for the test samples. This is a 1-D arrays of labels for Classification tasks or an array of output values for Regression tasks (STS-B).
            
        actual : array
            The actual values of the output for the test samples.
            
        topK : int
            Not used for this dataset
            
        confMat : Boolean
            For classification cases, this indicates whether the confusion matrix should be calculated. If the number of classes is more than 10, this argument is ignored and confusion matrix is not calculated. This is ignored for regression cases.

        expAcc : Boolean or None
            Ignored for regression cases. For classification cases:
                 
            * If this is a True, the expected accuracy and kappa values are also calculated. When the number of classes and/or number of evaluation samples is large, calculating expected accuracy can take a long time.
            * If this is False, the expected accuracy and kappa are not calculated.
            * If this is None (the default), then the expected accuracy and kappa are calculated only if number of classes does not exceed 10.
                                        
            **Note**: If confMat is True, then expAcc is automatically set to True.

        quiet : Boolean
            If False, it prints the test results.
                
        Returns
        -------
        list of tuples or dict
            For 'test' datasets, since the labels are unknown, this function returns a list of tuples like (index, predictedLabel) which can be used to make the 'tsv' files for submission to GLUE website. Otherwise, a dictionary of evaluation result values is returned.
        """
        taskLo = self.taskName.lower()
        if taskLo == 'sts-b':
            predicted = np.clip(predicted,0,5.0)    # A simple post processing for STS-B
        
        if self.dsName.lower()=='test':
            # In this case we don't have the actual labels. We create a list of tuples like (index, predictedLabel)
            # and return it. It can be used to make the 'tsv' files for submission to GLUE website.
            def predValToLabel(p):
                if taskLo in ['mnli-m', 'mnli-mm', 'snli', 'ax']: return ['contradiction', 'entailment', 'neutral'][p]
                if taskLo in ['qnli', 'rte']:                     return ['not_entailment', 'entailment'][p]
                if taskLo == 'sts-b':                             return float(p)
                return int(p)   # Otherwise: 'cola', 'mrpc', 'qqp', 'SST-2', 'wnli'
            
            return [ (i,predValToLabel(p)) for i,p in enumerate(predicted) ]
            
        if taskLo in ['cola', 'mrpc', 'qqp']:
            # Need to get confusion matrix for CoLA, MRPC, and QQP
            confMat, expAcc = True, True
            results = self.evaluateClassification(predicted, actual, 0, confMat, expAcc, quiet)
            confusion = results['confusionAP']
            tn = confusion[0][0]
            tp = confusion[1][1]
            fn = confusion[1][0]
            fp = confusion[0][1]
            
            if taskLo == 'cola':
                # Adding Matthews Correlation Coefficient for CoLA
                results['MCC'] = (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                results['csvItems'] += ['MCC']

                # To make sure we are calculating it correctly, un-remark these and test.
                # import sklearn.metrics
                # mcc2 = sklearn.metrics.matthews_corrcoef(actual, predicted)
                # if not quiet: print('MCC: %f'%(mcc2))
                if not quiet: print('MCC: %f'%(results['MCC']))
                
            else:
                recall = tp/(tp+fn)
                precision = tp/(tp+fp)
                results['F1'] = (2.0*recall*precision)/(recall+precision)
                results['csvItems'] += ['F1']
                if not quiet: print('F1: %f'%(results['F1']))

            return results
            
        elif taskLo == 'sts-b':
            # Adding Pearson Correlation for STS-B
            results = self.evaluateRegression(predicted, actual, quiet)
            
            pred1D, act1D = predicted.reshape((-1)), actual.reshape((-1))
            pearsonr = np.cov(pred1D,act1D)[0][1]/(np.std(pred1D)*np.std(act1D))
            results['PCorr'] = pearsonr
            results['csvItems'] += ['PCorr']
            
            if not quiet: print('Pearsons Correlation: %f'%(pearsonr))
            return results
            
        # Using accuracy for all other classification tasks (this is implemented in base)
        return self.evaluateClassification(predicted, actual, 0, confMat, expAcc, quiet)
