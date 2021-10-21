# Copyright (c) 2020 InterDigital AI Research Lab
r"""
This module contains the implementation of `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`_ dataset class for NLP Question-Answering tasks.
Use the ``SquadDSetUnitTest.py`` file in the ``UnitTest/Datasets`` folder to run the Unit Test of this implementation.

This implementation assumes that the following files exist in the 'dataPath' directory:

    * ``train-v1.1.json``: The training dataset for SQuAD version 1
    * ``dev-v1.1.json``: The evaluation dataset for SQuAD version 1
    * ``train-v2.0.json``: The training dataset for SQuAD version 2
    * ``dev-v2.0.json``: The evaluation dataset for SQuAD version 2
    
Note
----
    * When training, the actual answer text is not used. Only start and end tokens are included in the labels.
    * When evaluating, the start and end positions are not used and the answer text is used to compare with the predicted answer.
    * If there is a ``vocab.txt`` file in the 'dataPath' directory and no tokenizer is specified, this file is used to create a tokenizer.
    * The first time each JSON file is read, the tokenized information is saved in FNJ files. The FNJ files are used every time after that which makes the loading process faster.
    * While reading JSON files some of the samples are dropped because one of the following reasons:

      * The question length is too short (less than 4 tokens)
      * Questions with no answers in SQuAD version 1.
      * Answer tokens could not be found in the context. (See the method "shouldDrop")
  
**Dataset Stats**

    * Version 1:
    
    +--------------------+----------+------------+---------------------------------------+
    | Parameter          | Training | Evaluation | Comments                              |
    +====================+==========+============+=======================================+
    | Num Samples        | 87844    | 10833      | Total number of samples (segmented    |
    |                    |          |            |                                       |
    |                    |          |            | contexts counted multiple times).     |
    +--------------------+----------+------------+---------------------------------------+
    | Num Questions      | 87599    | 10570      | Total number of questions in the      |
    |                    |          |            |                                       |
    |                    |          |            | dataset. (Some questions ignored)     |
    +--------------------+----------+------------+---------------------------------------+
    | Num Questions Kept | 87451    | 10570      | Number of questions kept.             |
    +--------------------+----------+------------+---------------------------------------+
    | Num Answers        | 87844    | 35556      | Total Number of answers (Multiple     |
    |                    |          |            |                                       |
    |                    |          |            | answers for same question counted     |
    |                    |          |            |                                       |
    |                    |          |            | multiple times                        |
    +--------------------+----------+------------+---------------------------------------+
    | Num Contexts       | 18896    | 2067       | Total number of context paragraphs.   |
    +--------------------+----------+------------+---------------------------------------+
    | Num Titles         | 442      | 48         | Total number of subjects (titles)     |
    +--------------------+----------+------------+---------------------------------------+
    | Max Context Len    | 853      | 789        | Maximum length of context paragraphs. |
    +--------------------+----------+------------+---------------------------------------+
    | Max Question Len   | 61       | 38         | Maximum length of questions.          |
    +--------------------+----------+------------+---------------------------------------+
    | Num Impossible     | 0        | 0          | Number of questions with no answer    |
    +--------------------+----------+------------+---------------------------------------+
    | Max Num Answers    | 1        | 6          | Maximum number of answers for a       |
    |                    |          |            |                                       |
    |                    |          |            | question.                             |
    +--------------------+----------+------------+---------------------------------------+
    | Num Segmented      | 893      | 183        | Number of times a context paragraph   |
    |                    |          |            |                                       |
    |                    |          |            | was segmented because it was too      |
    |                    |          |            |                                       |
    |                    |          |            | long. This is based on the following  |
    |                    |          |            |                                       |
    |                    |          |            | segmentation params:                  |
    |                    |          |            |                                       |
    |                    |          |            | * maxSeqLen = 384                     |
    |                    |          |            | * stride = 128                        |
    |                    |          |            | * maxQuestionLen = 64                 |
    +--------------------+----------+------------+---------------------------------------+

    * Version 2:
    
    +--------------------+----------+------------+---------------------------------------+
    | Parameter          | Training | Evaluation | Comments                              |
    +====================+==========+============+=======================================+
    | Num Samples        | 131805   | 12232      | Total number of samples (segmented    |
    |                    |          |            |                                       |
    |                    |          |            | contexts are counted multiple times). |
    +--------------------+----------+------------+---------------------------------------+
    | Num Questions      | 130319   | 11873      | Total number of questions in the      |
    |                    |          |            |                                       |
    |                    |          |            | original dataset. (Some questions are |
    |                    |          |            |                                       |
    |                    |          |            | ignored)                              |
    +--------------------+----------+------------+---------------------------------------+
    | Num Questions Kept | 130184   | 11873      | Number of questions kept.             |
    +--------------------+----------+------------+---------------------------------------+
    | Num Answers        | 87074    | 20850      | Total Number of answers (Multiple     |
    |                    |          |            |                                       |
    |                    |          |            | answers forthe same question counted  |
    |                    |          |            |                                       |
    |                    |          |            | multiple times.                       |
    +--------------------+----------+------------+---------------------------------------+
    | Num Contexts       | 19035    | 1204       | Total number of context paragraphs.   |
    +--------------------+----------+------------+---------------------------------------+
    | Num Titles         | 442      | 35         | Total number of subjects (titles)     |
    +--------------------+----------+------------+---------------------------------------+
    | Max Context Len    | 853      | 789        | Maximum length of context paragraphs. |
    +--------------------+----------+------------+---------------------------------------+
    | Max Question Len   | 61       | 38         | Maximum length of questions.          |
    +--------------------+----------+------------+---------------------------------------+
    | Num Impossible     | 44731    | 6129       | Number of questions with no answer    |
    |                    |          |            |                                       |
    |                    |          |            | (segmented samples are counted        |
    |                    |          |            |                                       |
    |                    |          |            | multiple times)                       |
    +--------------------+----------+------------+---------------------------------------+
    | Max Num Answers    | 1        | 6          | Maximum number of answers for a       |
    |                    |          |            |                                       |
    |                    |          |            | question.                             |
    +--------------------+----------+------------+---------------------------------------+
    | Num Segmented      | 1373     | 210        | Number of times a context paragraph   |
    |                    |          |            |                                       |
    |                    |          |            | was segmented because it was too long.|
    |                    |          |            |                                       |
    |                    |          |            | This is based on the following        |
    |                    |          |            |                                       |
    |                    |          |            | segmentation params:                  |
    |                    |          |            |                                       |
    |                    |          |            | * maxSeqLen = 384                     |
    |                    |          |            | * stride = 128                        |
    |                    |          |            | * maxQuestionLen = 64                 |
    +--------------------+----------+------------+---------------------------------------+
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 05/20/2020    Shahab Hamidi-Rad       Created the file.
# 07/22/2020    Shahab Hamidi-Rad       Completed first version.
# 07/24/2020    Shahab Hamidi-Rad       Added support for H5 tokenized files. Printing info and Statistics.
# 07/29/2020    Shahab Hamidi-Rad       Added support for serial/parallel iteration. Completed the documentation.
# 08/04/2020    Shahab Hamidi-Rad       Added support for evaluation. Added "updateQuestionResults" method used
# 08/11/2020    Shahab Hamidi-Rad       Use FNJ files instead of H5 for the tokenized files. (See the fnjfile.py
#                                       for more info)
# 11/16/2020    Shahab Hamidi-Rad       Because of the new BERT implementation, we don't need the "noPadLen" to be
#                                       included for every sample in the dataset. So, from now on, each sample in the
#                                       dataset is specified by a tuple of (tokenIds, tokenTypes)
# 10/11/2021    Shahab Hamidi-Rad       Added support for downloading datasets.
# **********************************************************************************************************************
import json
import numpy as np
import os, time
from .base import BaseDSet
from ..printutils import myPrint
from ..fnjfile import loadFNJ, saveFNJ

# **********************************************************************************************************************
class SquadDSet(BaseDSet):
    r"""
    This class implements the SQuAD dataset.
    """
    classNames = None
    numClasses = 0

    maxSeqLen = 384
    stride = 128
    maxQuestionLen = 64
    maxAnswerLength = 30
    
    # ******************************************************************************************************************
    def __init__(self, dsName='Train', dataPath=None, batchSize=8, version=2, tokenizer=None, numWorkers=0):
        r"""
        Constructs a `SquadDSet` instance. This can be called directly or via `makeDatasets` class method.
        
        Parameters
        ----------
        dsName : str, optional
            The name of the dataset. It can be one of "Train" or "Test".

        dataPath : str, optional
            The path to the directory where the dataset files are located.
            
        batchSize : int, optional
            The default batch size used in the "batches" method.
        
        version : int, optional
            The SQuAD version of the dataset. It can be 1 or 2.
        
        tokenizer : Tokenizer object, optional
            The tokenizer used to tokenize the text info in the dataset files.
            
        numWorkers : int, optional
            If numWorkers is more than zero, “numWorkers” worker threads are created to process and prepare future batches in parallel.
        """
        if dataPath is None:
            dataPath = '/data/SQuAD/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)
        assert tokenizer is not None, "'tokenizer' cannot be None!"
        
        self.version = version
        self.tokenizer = tokenizer
        super().__init__(dsName, dataPath, None, None, batchSize, numWorkers)
        self.sampleShape = (self.maxSeqLen,)        # Length of question+context (padded to "maxSeqLen")
        self.labelShape = (2,)                      # Start and End Indexes
        self.__class__.evalMetricName = 'Accuracy'
            
    # ******************************************************************************************************************
    @classmethod
    def download(cls, dataFolder=None):
        r"""
        This class method can be called to download the SQuAD dataset files.
        
        Parameters
        ----------
        dataFolder: str
            The folder where dataset files are saved. If this is not provided, then
            a folder named "data" is created in the home directory of the current user and the
            dataset folders and files are created there. In other words, the default data folder
            is ``~/data``
        """
        BaseDSet.download("SQuAD", ['SQuAD.zip'], dataFolder)

    # ******************************************************************************************************************
    def getSamplesAndLabels(self, fileName):
        r"""
        This method loads the samples and labels form the dataset files and populates the statistics information in the "stats" dictionary. The first time a JSON file is read, the tokenized information is saved to an FNJ file. This FNJ file is used every time after that. This makes the loading process much faster.
        
        Parameters
        ----------
        fileName : str
            The FNJ or JSON dataset file name containing the sample/label information.
        """
        numGoodQuestions = 0    # Total number of questions included in the samples
        maxContextLen = 0
        maxQLen = 0             # Max length of questions in the dataset (Different from maxQuestionLen)
        maxNumAnswers = 0
      
        def shouldDrop(answerToksFromContext, answerToks):
            if answerToksFromContext[0] == answerToks[0]:   return False
            if len(answerToksFromContext)==1 and len(answerToks)==1:
                if answerToks[0].isnumeric():
                    if answerToksFromContext[0].isnumeric():            return True     # Example: '3' and '33' not ok
                    if answerToksFromContext[0].find(answerToks[0])==0: return False    # Example: '41' and '41st' ok
                        
                if len(answerToks[0])<3:    return True             # Example: 'no' & 'notes', 'h' & 'hydrogen' not ok
                if answerToksFromContext[0].find(answerToks[0])==0: return False        # Example: 'six' & 'sixth' ok
            
            return True

        self.titles = []
        self.contextIdxToTitleIdx = []
        self.contexts = []
        self.contextsTokIds = []
        self.contextsTokSpans = []
        
        self.questions = []
        self.questionsTokIds = []
        self.questionIdToIdx = {}
        self.questionIdxToId = []
        self.questionIdxToContextIdx = []

        self.exampleContextIdxs = []
        self.exampleQuestionIdxs = []
        self.exampleAnswerSpans = []
        self.exampleAnswerTexts = []

        examples = []
        if os.path.exists(fileName + '.fnj'):
            rootDic = loadFNJ(fileName + '.fnj')
            self.titles = rootDic['titles']
            self.contextIdxToTitleIdx = rootDic['contextIdxToTitleIdx'].tolist()
            
            self.contexts = rootDic['contexts']
            self.contextsTokIds = [x.tolist() for x in rootDic['contextsTokIds']]
            self.contextsTokSpans = [x.tolist() for x in rootDic['contextsTokSpans']]

            self.questions = rootDic['questions']
            self.questionsTokIds = [x.tolist() for x in rootDic['questionsTokIds']]
            self.questionIdToIdx = rootDic['questionIdToIdx']
            self.questionIdxToId = rootDic['questionIdxToId']
            self.questionIdxToContextIdx = rootDic['questionIdxToContextIdx'].tolist()

            self.exampleContextIdxs = rootDic['exampleContextIdxs'].tolist()
            self.exampleQuestionIdxs = rootDic['exampleQuestionIdxs'].tolist()
            self.exampleAnswerSpans = rootDic['exampleAnswerSpans']
            self.exampleAnswerTexts = rootDic['exampleAnswerTexts']

            numGoodQuestions = len(self.questions)    # Total number of questions included in the samples
            maxContextLen = max( [len(x) for x in self.contextsTokIds] )
            maxQLen = max( [len(x) for x in self.questionsTokIds] )
            maxNumAnswers = max( [len(x) for x in self.exampleAnswerTexts] )

        elif os.path.exists(fileName + '.json'):
            with open(fileName+'.json', "r", encoding="utf-8") as jsonFile:
                data = json.load(jsonFile)["data"]
            for item in data:
                titleIdx = len(self.titles)
                self.titles += [ item['title'] ]
                for paragraph in item['paragraphs']:
                    context = paragraph['context']
                    contextTokens, contextTokSpans = self.tokenizer.tokenize(context)
                    contextTokIds = self.tokenizer.toIds(contextTokens)
                    if len(contextTokIds) > maxContextLen: maxContextLen = len(contextTokIds)
                    contextIdx = len(self.contextsTokIds)
                    self.contexts += [ context ]
                    self.contextsTokIds += [ contextTokIds ]
                    self.contextsTokSpans += [ contextTokSpans ]
                    self.contextIdxToTitleIdx += [ titleIdx ]

                    for qa in paragraph['qas']:
                        qaId = qa['id']
                        question = qa['question']
                        if 'is_impossible' not in qa:   impossible = False
                        else:                           impossible = qa['is_impossible']
                        
                        questionIdx = len(self.questions)
                        questionTokIds, _ = self.tokenizer.encode(question)

                        answerStart = -1
                        answerInfo = []

                        if impossible:
                            self.questions += [ question ]
                            self.questionsTokIds += [ questionTokIds ]
                            self.questionIdxToId += [ qaId ]
                            self.questionIdToIdx[ qaId ] = questionIdx
                            self.questionIdxToContextIdx += [ contextIdx ]
                            
                            self.exampleContextIdxs += [ contextIdx ]
                            self.exampleQuestionIdxs += [ questionIdx ]
                            self.exampleAnswerSpans += [ [[0,0]] ]
                            self.exampleAnswerTexts += [ [""] ]

                            numGoodQuestions += 1
                            continue
                        
                        # Not impossible:
                        answers = qa['answers']
                        if self.isTraining:
                            assert len(answers)==1, "Number of answers(%d) must be 1 in training mode!"%(len(answers))
                            
                        for answer in answers:
                            answerText = answer['text']
                            answerStart = answer['answer_start']
                            answerEnd = answerStart+len(answerText)

                            if self.isTraining:
                                # For training we need start and end positions as labels.
                                assert answerStart<len(context)
                                assert answerEnd<=len(context)
                                
                                # The answer must be in the context:
                                answerTokens, _ = self.tokenizer.tokenize(answerText)
                                answerTokensFromContext, _ = self.tokenizer.tokenize(context[answerStart:answerEnd])
                                assert "".join(answerTokens) == "".join(answerTokensFromContext)

                                contextIdsBeforeAnswer, _ = self.tokenizer.tokenize(context[:answerStart])
                                startTokIndex = len(contextIdsBeforeAnswer)
                                endTokIndex = startTokIndex + len(answerTokens)-1

                                if shouldDrop(contextTokens[startTokIndex:endTokIndex+1], answerTokens):    continue
                                answerInfo += [(startTokIndex, endTokIndex, answerText)]
                            else:
                                # For evaluation, we only need the actual text, but we add the start and end char
                                # indexes as read from the dataset
                                answerInfo += [ (answerStart, answerEnd, answerText) ]

                        if len(answerInfo) == 0:    continue
                        self.questions += [ question ]
                        self.questionsTokIds += [ questionTokIds ]
                        self.questionIdxToId += [ qaId ]
                        self.questionIdToIdx[ qaId ] = questionIdx
                        self.questionIdxToContextIdx += [ contextIdx ]

                        self.exampleContextIdxs += [ contextIdx ]
                        self.exampleQuestionIdxs += [ questionIdx ]
                        self.exampleAnswerSpans += [ [[a[0],a[1]] for a in answerInfo] ]
                        self.exampleAnswerTexts += [ [a[2] for a in answerInfo] ]

                        numGoodQuestions += 1
                        if len(answerInfo) > maxNumAnswers:     maxNumAnswers = len(answerInfo)
                        if len(questionTokIds) > maxQLen:       maxQLen = len(questionTokIds)

            myPrint('saving tokenized info to "%s" ... '%(fileName+'.fnj'), False)
            self.saveData(fileName+'.fnj')
            myPrint('Done!')

        numSegmented = 0
        numAnswers = 0
        numImpossible = 0
        self.samples = []
        self.labels = []
        self.sampleInfo = []
        self.qaidToSampleIdx = {}
        
        numExamples = len( self.exampleContextIdxs )
        for e in range(numExamples):
            contextIdx = self.exampleContextIdxs[e]
            questionIdx = self.exampleQuestionIdxs[e]
            answerSpans = self.exampleAnswerSpans[e]
            answerTexts = self.exampleAnswerTexts[e]
            qaId = self.questionIdxToId[questionIdx]
            
            questionTokIds = self.questionsTokIds[questionIdx][0:self.maxQuestionLen]
            if len(questionTokIds) < 4:    continue    # Drop very short questions
            contextTokIds = self.contextsTokIds[ contextIdx ]
            contextLen = len(contextTokIds)
            maxSegLen = self.maxSeqLen - len(questionTokIds) - 3
                
            # If contextTokIds is longer than "maxSegLen", then we need to break it down to "segments".
            numSegmented += 1 if contextLen>maxSegLen else 0
            remainingContextLen = contextLen
            segStart = 0
            segments = []
            
            # For evaluation, some tokens appear in more than one segment. This means the same token can have
            # different probability for being start or end of answer when each one of the segments is fed to
            # the model.
            # We only consider the probability for a token in a segment if the token has its maximum contextScore
            # in this segment.
            # The contextScore of token at position "t" in a segment from "segStart" to "segEnd" is defined as:
            #       min(t-segStart, segEnd-t)+0.01*(segLen)
            contextScores = []
            while 1:
                segLen = min(remainingContextLen, maxSegLen)
                segEnd = segStart + segLen - 1
                segments += [ (segStart, segEnd) ]
                contextScores += [ [ min(t-segStart, segEnd-t) + .01*(segLen) for t in range(contextLen) ] ]
                if remainingContextLen<=maxSegLen:  break
                remainingContextLen -= self.stride
                segStart += self.stride

            contextOffset = 2 + len(questionTokIds)
            tokenInMaxContext = ((np.max(contextScores,0)-contextScores)==0).tolist()
            for s,(segStart, segEnd) in enumerate(segments):
                # Each sample is a tuple of the form (tokenIds, tokenTypes) where:
                #   tokenIds:   A list of integer token IDs with length fixed to "maxSeqLen":
                #               [CLS] + questionTokenIds + [SEP] + segmentTokenIds + [SEP] + [PAD] + ... + [PAD]
                #   tokenTypes: A list of integer values indicating token types for each token in tokenIds. We
                #               use 0 for question tokens and 1 for context (or this segment of context). 0 is
                #               also used for the padding tokens.
                sample = self.tokenizer.packTokenIds(questionTokIds, contextTokIds[segStart:segEnd+1], self.maxSeqLen)
                
                if self.isTraining:
                    # For training, there is only one answer. So, the label is just start and end token of the answer.
                    label = (0, 0)
                    answerStart, answerEnd = answerSpans[0]
                    answerText = answerTexts[0]
                    impossible = False
                    if answerStart == 0 and answerEnd == 0 and answerText == "":    impossible = True
                    elif (answerStart >= segStart) and (answerEnd <= segEnd):
                        # The answer is completely in this segment
                        label = (answerStart-segStart+contextOffset, answerEnd-segStart+contextOffset)
                    else:
                        # If the answer is not found in this segment, this sample is impossible
                        impossible = True
                    
                    if impossible:
                        if self.version<2:    continue    # Drop impossible samples for version 1
                        numImpossible += 1
                    else:
                        numAnswers += 1
                else:
                    # For evaluation, the label has the actual text of answers and the a flag for each token
                    # in the segment that indicates whether the token has the maximum contextScore in this segment.
                    answerTexts = [answerText for answerText in answerTexts if answerText!=""]
                    label = answerTexts
                    
                    if len(answerTexts)==0:
                        if self.version<2:    continue    # Drop impossible samples for version 1
                        numImpossible += 1
                    else:
                        numAnswers += len(answerTexts)

                self.qaidToSampleIdx[qaId] = self.qaidToSampleIdx.get(qaId,[]) + [len(self.samples)]
                self.samples += [ sample ]
                self.labels += [ label ]
                self.sampleInfo += [(questionIdx, segStart, segEnd, len(segments), tokenInMaxContext[s][segStart:segEnd+1])]

        self.stats = {
            "NumQuestions":     len(self.questions),
            "NumGoodQuestions": numGoodQuestions,
            "NumAnswers":       numAnswers,
            "NumContexts":      len(self.contextsTokIds),
            "NumTitle":         len(self.titles),
            "MaxContextLen":    maxContextLen,              # num tokens
            "MaxQuestionLen":   maxQLen,                    # num tokens
            "NumImpossible":    numImpossible,
            "MaxNumAnswers":    maxNumAnswers,
            "NumSegmented":     numSegmented,
            "NumSamples":       len(self.samples),
            }

    # ******************************************************************************************************************
    def saveData(self, fileName):
        r"""
        This method saves the tokenized information into a FNJ file. Loading the tokenized information is much faster than reading and tokenizing JSON files.
        
        Parameters
        ----------
        examples : list of tuples
            Contains all the refined and tokenized information read from the JSON file before segmentation.
        
        fileName: str
            The name of destination FNJ file.
            
        maxContextLen: int
            The maximum length of tokenized context paragraphs in the JSON file.
    
        maxQLen: int
            The maximum length of tokenized questions in the JSON file.

        maxNumAnswers: int
            The maximum number of questions available for questions in the JSON file.
        """
        rootDic = {
                    'titles':                   self.titles,
                    'contextIdxToTitleIdx':     np.array(self.contextIdxToTitleIdx, dtype=np.uint16),

                    'contexts':                 self.contexts,
                    'contextsTokIds':           [ np.array(x,dtype=np.uint16) for x in self.contextsTokIds ],
                    'contextsTokSpans':         [ np.array(x,dtype=np.uint16) for x in self.contextsTokSpans ],
                    
                    'questions':                self.questions,
                    'questionsTokIds':          [ np.array(x,dtype=np.uint16) for x in self.questionsTokIds],
                    'questionIdToIdx':          self.questionIdToIdx,
                    'questionIdxToId':          self.questionIdxToId,
                    'questionIdxToContextIdx':  np.array(self.questionIdxToContextIdx, dtype=np.uint16),

                    'exampleContextIdxs':       np.array(self.exampleContextIdxs, dtype=np.uint16),
                    'exampleQuestionIdxs':      np.array(self.exampleQuestionIdxs, dtype=np.uint32),
                    'exampleAnswerSpans':       self.exampleAnswerSpans,
                    'exampleAnswerTexts':       self.exampleAnswerTexts,
                  }
        saveFNJ(fileName, rootDic)

    # ******************************************************************************************************************
    def loadSamples(self):
        r"""
        This function is called by the constructor of the base dataset class to load the samples and labels of this dataset based on `dsName`, `dataPath`, and 'version' properties of this class.
        
        Note
        ----
        The dsName "Valid" cannot be used here. A validation dataset should be created using the `makeDatasets` method or using the `split` method on an existing training dataset.
        """

        if self.isTraining:
            fileName = 'train-v2.0' if self.version==2 else 'train-v1.1'
            self.getSamplesAndLabels(self.dataPath+fileName)
        elif 'test' in self.dsName.lower():
            fileName = 'dev-v2.0' if self.version==2 else 'dev-v1.1'
            self.getSamplesAndLabels(self.dataPath+fileName)

        else:
            raise ValueError("Unsupported dataset name \"%s\"!"%(self.dsName))

        
    # ******************************************************************************************************************
    @classmethod
    def makeDatasets(cls, dsNames='Train,Test', batchSize=8, dataPath=None, version=2, tokenizer=None, numWorkers=0):
        r"""
        This class method creates several datasets in one-shot as specified by `dsNames` parameter.
        
        Parameters
        ----------
        dsNames : str
            A combination of the following:
            
            * **"Train"**: Create the training dataset.
            * **"Test"**: Create the test dataset.
              
        batchSize : int
            The batchSize used for all the datasets created.
        
        dataPath : str
            The path to the directory where the dataset files are located.
            
        version : int
            The version of SQuAD dataset (1 or 2)

        tokenizer : Tokenizer object
            The tokenizer used by all created datasets. If this is None, and there is a "vocab.txt" file in the "dataPath", this method tries to create a tokenizer using "vocab.txt" as its vocabulary.
            
        Returns
        -------
        Up to 2 SquadDSet objects
            Depending on the number of items specified in the `dsNames`, it returns one or two SquadDSet objects. The returned values have the same order as they appear in the `dsNames` parameter.
        """
        if dataPath is None:
            dataPath = '/data/SQuAD/'
            if os.path.exists(dataPath) == False: dataPath = os.path.expanduser("~") + dataPath
        assert os.path.exists(dataPath), "Could not find the dataset folder '%s'"%(dataPath)

        if tokenizer is None:
            if os.path.exists(dataPath+'vocab.txt'):
                myPrint('Initializing tokenizer from "%s" ... '%(dataPath+'vocab.txt'), False)
                from ..textio import Tokenizer
                tokenizer = Tokenizer(dataPath+'vocab.txt')
                myPrint('Done. (Vocab Size: %d)'%(tokenizer.vocabSize))
            else:
                raise ValueError("A tokenizer is needed for the SQuAD dataset to work properly!")

        validSource = None
        dsStrs = dsNames.split(',')
        retVals = []
        for dsStr in dsStrs:
            retVals += [ cls(dsStr, dataPath, batchSize, version, tokenizer, numWorkers) ]

        cls.postMakeDatasets()
        if len(retVals)==1: return retVals[0]
        return retVals

    # ******************************************************************************************************************
    @classmethod
    def printDsInfo(cls, trainDs=None, testDs=None):
        r"""
        This class method prints information about given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : SquadDSet, optional
            The training dataset.
            
        testDs : SquadDSet, optional
                The test dataset.
        """
        
        if trainDs is not None:     dataPath, version = trainDs.dataPath, trainDs.version
        elif testDs is not None:    dataPath, version  = testDs.dataPath, testDs.version

        print('%s Dataset Info:'%(cls.__name__))
        if dataPath is not None:
            print('    Dataset Location ............................... %s'%(dataPath))
        if trainDs is not None:
            print('    Number of Training Samples ..................... %d'%(trainDs.numSamples))
        if testDs is not None:
            print('    Number of Test Samples ......................... %d'%(testDs.numSamples))
        if (trainDs is not None) and trainDs.numWorkers>0:
            print('    Number of Worker Threads ....................... %d'%(trainDs.numWorkers))
        print('    Dataset Version ................................ %s'%(str(version)))
        print('    Max Seq. Len ................................... %d'%(cls.maxSeqLen))

    # ******************************************************************************************************************
    @classmethod
    def printStats(cls, trainDs=None, testDs=None):
        r"""
        This class method prints dataset statistics for the given set of datasets in a single table.
        
        Parameters
        ----------
        trainDs : SquadDSet, optional
            The training dataset.
            
        testDs : SquadDSet, optional
                The test dataset.
        """
        #             | 12345678901234567890 |
        sep =    '    +----------------------+'
        rowStr = '    | Parameter            |'
        
        all = 0
        if trainDs is not None:
            #           123456789012 |
            sep +=    '--------------+'
            rowStr += ' Training     |'

        if testDs is not None:
            #           123456789012 |
            sep +=    '--------------+'
            rowStr += ' Test         |'

        print(sep)
        print(rowStr)
        print(sep)
        keys = testDs.stats.keys() if trainDs is None else trainDs.stats.keys()
        for paramName in keys:
            rowStr = '    | %-20s |'%(paramName)
            if trainDs is not None:     rowStr += ' %-12d |'%(trainDs.stats[paramName])
            if testDs is not None:      rowStr += ' %-12d |'%(testDs.stats[paramName])
            print( rowStr )
       
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
        samples : tuple of 3 items:
        
            * batchTokenIds:    2D list of integer tokenId values for 'n' sequences (where 'n' is current batch size). Each sequence (row) contains 'maxSeqLen' tokens including tokens for the question, the context, and paddings
            * batchTokenTypes:  2D list of integers one for each tokenId in "batchTokenIds". '0' is used for question tokens and '1' for context (or a segment of context). '0' is also used for the padding tokens.
            
        labels : tuple of 2 items
            For training mode, the label contains the following lists:
            
            * batchStartPos: For each sample in the batch, this contains the position of the first token of ground-truth answer.
            * batchStartPos: For each sample in the batch, this contains the position of the last token of ground-truth answer.
                               
            For evaluation mode, the label contains the following lists:
            
            * batchAnswerTexts: For each sample in the batch, this is a list of text strings containing the possible answers to the questions.
            * batchTokenInMaxContext: For each sample in the batch, this is a boolean list. Each item in the list indicates whether the corresponding token in the sequence is in its maximum context. This is used when the answer may appear in different segments of the same context. We only should consider the results when the token is in its max context.
        """
        
        batchSampleIdxs, batchTokenIds, batchTokenTypes = [], [], []
        batchStartIdxs, batchEndIdxs, batchAnswerTexts = [], [], []
        for i in batchIndexes:
            tokenIds, tokenTypes = self.samples[ i ]
            batchSampleIdxs += [ i ]
            batchTokenIds += [ tokenIds ]
            batchTokenTypes += [ tokenTypes ]
            
            if self.isTraining:     # For training mode, the 2 label components are startPos and endPos.
                startIdx, endIdx = self.labels[i]
                batchStartIdxs += [ startIdx ]
                batchEndIdxs += [ endIdx ]
            else:                   # For evaluation, the label for each sample is a list of actual answer texts
                batchAnswerTexts += [ self.labels[i] ]
        
        if self.isTraining:
            return (batchSampleIdxs, batchTokenIds, batchTokenTypes), (batchStartIdxs, batchEndIdxs)
        
        return (batchSampleIdxs, batchTokenIds, batchTokenTypes), batchAnswerTexts

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
            If true, instead of calculating all the results, just calculates the main metric of the dataset and returns that. This is mostly used during the training at the end of each epoch. The main metric for the SQuAD dataset is the exact match accuracy.
            
            Otherwise, if this is False (the default), the full results are calculated and a dictionary of all results is returned.

        **kwargs : dict
            This contains some additional task specific arguments. Here is
            a list of what can be included in this dictionary.
            
                * **maxSamples (int)**: The max number of samples from this dataSet to be processed for the evaluation of the model. If not specified, all samples are used (default behavior).

                * **jsonFile (str)**: If specified, this is the name of JSON file that is created by this function.
                    
        Returns
        -------
        If returnMetric is True, the actual value of dataset's main metric is returned.
        Otherwise, this function returns a dictionary containing the results of the evaluation process.
        """
        maxSamples=kwargs.get('maxSamples', None)
        jsonFile=kwargs.get('jsonFile', None)

        t0 = time.time()
        if maxSamples is None:  maxSamples = self.numSamples
        if batchSize is None:   batchSize = self.batchSize
        quietProcess = quiet or returnMetric
        totalSamples = 0
    
        questionResults = [ ([], [], []) ] * len(self.questions)
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

            sampleIdxs, _, _ = batchSamples
            totalSamples += len(sampleIdxs)
            if batchSize == 1: t0 = time.time()
            batchStartLogits, batchEndLogits = model.inferBatch( batchSamples, returnProbs=True )
            for i, sampleIdx in enumerate(sampleIdxs):
                self.updateQuestionResults(questionResults, sampleIdx, batchStartLogits[i], batchEndLogits[i])
        
            if b>0: totalTime +=(time.time()-t0)    # Do not count the first sample
            
        if not quietProcess:
            if batchSize==1:
                myPrint('\r  Processed %d Samples. (Time Per Sample: %.2f ms)%30s\n'%(totalSamples, (1000.0*totalTime)/(maxSamples-1),' '))
            else:
                myPrint('\r  Processed %d Samples. (Time: %.2f Sec.)%30s\n'%(totalSamples, time.time()-t0,' '))

        predictions = {}
        for questionIdx, (answerText, answerProb, noAnswerProb) in enumerate(questionResults):
            qaId = self.questionIdxToId[questionIdx]
            if type(answerText) != str:
                if maxSamples == self.numSamples:
                    assert type(answerText) == str, "Some segments of Question '%s' were not processed!"%(qaId)
                else:
                    # If maxSamples is smaller than all samples, then it is ok to have incomplete questions.
                    # So we just ignore them here.
                    continue
            predictions[qaId] = answerText
            
        evalResults = self.evaluate(predictions, quiet=quietProcess)

        if jsonFile is not None:
            if not quietProcess:  myPrint('Saving predictions to "predictions.json" ... ', False)
            import json
            with open(jsonFile, 'w') as jf:
                json.dump(predictions, jf)
            if not quietProcess:  myPrint('Done.\n')

        if returnMetric: return evalResults['exact']
        return evalResults

    # ******************************************************************************************************************
    def evaluate(self, predictions, noAnswerProbThreshold=1.0, quiet=False):
        r"""
        Returns information about evaluation results based on the "predicted" values. This function is usually called by the ``evaluateModel`` function. But it can also be called when we have a the prediction results as a dictionary.
        
        If you want to evaluate the prediction results from a JSON file, the name of JSON file can also be passed in prediction.
        
        Parameters
        ----------
        predictions : dict or str
            
            * If ``predictions`` is a dictionary, it should have the text string for each question Id in this dataset. In other words, the keys are the question IDs in this dataset and the values are the actual answer text string predicted by the model.
            
            * If ``predictions`` is a str, it should be the name of a JSON file containing the prediction results. The JSON file should contain a dictionay with a format as explained above.
                        
        noAnswerProbThreshold : float
            If the predicted probability of not having an answer for a question is more than this value, then we assume that the prediction is no-answer. In this case we consider this an exact match if the ``impossible`` flag for this question is set in the dataset and a mismatch otherwise.

        quiet : Boolean
            If False, it prints the test results. The printed information includes the Confusion matrix and accuracy information for Classification problems and MSE, RMS, MAE, and PSNR for Regression problems.
                
        Returns
        -------
        dict
            A dictionary of the results information. Here is a list of items in the results dictionary:
            
            * **exact**: The exact match accuracy. A float number between 0 and 1.
            * **f1**: The F1 value which is calculated based on how similar the predicted and the ground-truth answer texts are.
            * **numQuestions**: Total number of questions involved in the evaluation
            * **hasAnsExact**: The exact match accuracy for the questions that actually have an answer (Not impossible)
            * **hasAnsF1**: The F1 value for the questions that actually have an answer (Not impossible)
            * **numHasAns**: The number of questions that have an answer (Not Impossible) involved in the evaluation
            * **noAnsExact**: The exact match accuracy for the impossible questions.
            * **noAnsF1**: The F1 value for the impossible questions.
            * **numNoAns**: The number of impossible questions involved in the evaluation
            * **csvItems**: A list of evaluation metrics that will be included in the CSV file when performing a parameter search.
        """
        import re
        import collections
        # predictions is a dictionary (or name of a JSON file containing a dictionary) of
        #   {<qaId>: <answerText>} or {<qaId>: (<answerText>, <noAnsProb>)}
        if type(predictions)==str:
            with open(self.dataPath + predictions) as predFile:
                predictions = json.load(predFile)

        def normalize(answerStr):
            punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            answerStr = ''.join(char for char in answerStr.lower() if char not in punc)
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            answerStr = re.sub(regex, ' ', answerStr)
            answerStr = ' '.join(answerStr.split())
            return answerStr

        sumF1 =     [ 0.0, 0.0, 0.0 ]   # all, hasAns, noAns
        sumExact =  [ 0.0, 0.0, 0.0 ]   # all, hasAns, noAns
        numNoAns = 0
        for qid in self.qaidToSampleIdx:
            sampleIdx = self.qaidToSampleIdx[qid][0]
            answerStrs = self.labels[sampleIdx]

            normalizedAnswers = []
            for answerStr in answerStrs:
                answerStr = normalize(answerStr)
                if len(answerStr)>0:   normalizedAnswers += [answerStr]
            impossible = (len(normalizedAnswers)==0)
            if impossible:   normalizedAnswers = ['']
            numNoAns += 1 if impossible else 0

            if qid not in predictions:
                if not quiet:   myPrint("Warning: No prediction for question Id '%s'!"%(qid), color='yellow')
                continue

            noAnswerProb = 0.0
            if type(predictions[qid]) == tuple: predAnswerStr, noAnswerProb = predictions[qid]
            else:                               predAnswerStr = predictions[qid]
            predAnswerStr = normalize(predAnswerStr)
            
            exactMatch = 0.0
            f1 = 0
            for answerStr in normalizedAnswers:
                if answerStr == predAnswerStr:  exactMatch = 1.0
                
                answerTokens = answerStr.split()
                predAnswerTokens = predAnswerStr.split()
                if len(answerTokens)==0:        ansF1 = 1 if len(predAnswerStr)==0 else 0
                elif len(predAnswerStr)==0:     ansF1 = 0
                else:
                    # Get a Counter (dictionary) of {<tok>: <count>} where count is the number of times the
                    # token "tok" appeared in both "answerTokens" and "predAnswerTokens".
                    commonCounts = collections.Counter(answerTokens) & collections.Counter(predAnswerTokens)
                    numCommonTokens = sum(commonCounts.values())     # Total number of common tokens
                    if numCommonTokens == 0:    ansF1 = 0
                    else:
                        precision = float(numCommonTokens)/len(predAnswerTokens)
                        recall = float(numCommonTokens)/len(answerTokens)
                        ansF1 = (2 * precision * recall) / (precision + recall)

                if ansF1>f1:    f1 = ansF1

            if noAnswerProb > noAnswerProbThreshold:
                exactMatch = 1.0 if impossible else 0.0
                f1 = 1.0 if impossible else 0.0
            
            sumF1[0] += f1
            sumExact[0] += exactMatch
            
            sumF1[ 2 if impossible else 1] += f1
            sumExact[ 2 if impossible else 1] += exactMatch

        numQuestions = len(self.qaidToSampleIdx)
        results = {
            'exact':        100.0 * sumExact[0] / numQuestions,
            'f1':           100.0 * sumF1[0] / numQuestions,
            'numQuestions': numQuestions,
            'hasAnsExact':  100.0 * sumExact[1] / (numQuestions-numNoAns),
            'hasAnsF1':     100.0 * sumF1[1] / (numQuestions-numNoAns),
            'numHasAns':    numQuestions-numNoAns,
            'noAnsExact':   0 if numNoAns==0 else 100.0 * sumExact[2] / numNoAns,
            'noAnsF1':      0 if numNoAns==0 else 100.0 * sumF1[2] / numNoAns,
            'numNoAns':     numNoAns,
            
            'csvItems':     ['exact','f1','numQuestions',
                             'hasAnsExact','hasAnsF1','numHasAns',
                             'noAnsExact','noAnsF1','numNoAns'],
        }
        
        if not quiet:
            print('    Exact Match: %-.3f'%(results['exact']))
            print('    f1:          %-.3f\n'%(results['f1']))

        return results

    # ******************************************************************************************************************
    def updateQuestionResults(self, questionResults, sampleIdx, startLogits, endLogits,
                              numTopLogits = 20, maxAnswerLen = 30, noAnswerScoreDiffThreshold=0):
        questionIdx, _, _, numSeg, _ = self.sampleInfo[sampleIdx]
        
        otherSegmentsInfo, _ , _ = questionResults[questionIdx]
        allSegmentsInfo = otherSegmentsInfo+[(sampleIdx, startLogits, endLogits)]
        if len(allSegmentsInfo) < numSeg:
            # We don't have all the segments for this question yet.
            questionResults[questionIdx] = (allSegmentsInfo, None, None)
            return

        contextIdx = self.questionIdxToContextIdx[ questionIdx ]
        contextTokSpans = self.contextsTokSpans[ contextIdx ]

        qaId = self.questionIdxToId[questionIdx]
        doPrint = False
        if doPrint: print("*"*100)
        if doPrint: print("Context:\n", self.contexts[ contextIdx ])
        if doPrint: print("Question:\n", self.questions[ questionIdx ])
        if doPrint: print("numSeg:", numSeg)

        # Now we have all the segments of this question. Let's find all feasible
        # answers first:
        feasibleAnswers = []
        minNoAnswerScore = 10000000
        for sampleIdx, startLogits, endLogits in allSegmentsInfo:
            _, segStart, segEnd, _, tokenInMaxContext = self.sampleInfo[sampleIdx]
            segLen = segEnd - segStart + 1
        
            minIdx = len( self.questionsTokIds[questionIdx] ) + 2
            maxIdx = minIdx + segLen
            
            # Get indexes of top logits for start and end from token indexes in minIdx to maxIdx
            topStartIdxs = minIdx + np.argsort(-startLogits[minIdx:maxIdx])[:numTopLogits]  # Sorted descending
            topEndIdxs = minIdx + np.argsort(-endLogits[minIdx:maxIdx])[:numTopLogits]      # Sorted descending

            if doPrint: print("segStart, segEnd:", segStart, segEnd)
            if doPrint: print("topStartIdxs:\n", topStartIdxs)
            if doPrint: print("topEndIdxs:\n", topEndIdxs)

            if doPrint: print("tokenInMaxContext:\n", tokenInMaxContext)
            if doPrint: print("answers:\n", self.labels[sampleIdx] )
            if doPrint: print("minIdx, maxIdx:", minIdx, maxIdx)

            # Filter out infeasible start/end indexes.
            for startIdx in topStartIdxs:
#                if (startIdx<minIdx) or (startIdx>maxIdx):              continue
                if tokenInMaxContext is not None:
                    if tokenInMaxContext[ startIdx - minIdx ]==False:   continue
                for endIdx in topEndIdxs:
#                    if (endIdx<startIdx) or (endIdx>maxIdx):            continue
                    if (endIdx<startIdx):                   continue
                    if (endIdx-startIdx+1)>maxAnswerLen:    continue
                    answerScore = startLogits[startIdx] + endLogits[endIdx]
                    feasibleAnswers += [(startIdx+segStart, endIdx+segStart, answerScore)]

            noAnswerScore = startLogits[0] + endLogits[0]
            if noAnswerScore < minNoAnswerScore:    minNoAnswerScore = noAnswerScore

        # Always add "no-answer" to the feasible answers
        feasibleAnswers += [(0, 0, minNoAnswerScore)]


        # Sort feasible answers in the order of descending scores
        feasibleAnswers = sorted( feasibleAnswers, key=lambda x: x[2], reverse=True)
        if doPrint: print("feasibleAnswers:\n", feasibleAnswers)

        # Now keep the top "numTopLogits" answers:
        bestAnswers = {}
        scores = []
        bestAnswer = ""
        context = self.contexts[ contextIdx ]
        for startIdx, endIdx, answerScore in feasibleAnswers:
            if startIdx>0:  # Not a "no answer" case
                startCharIdx = contextTokSpans[startIdx - minIdx][0]
                endCharIdx = contextTokSpans[endIdx - minIdx][1]
                answerText = context[ startCharIdx : endCharIdx ]
                answerText = " ".join(answerText.split())           # A simple normalization of white spaces.
                if bestAnswer == "":    bestAnswer = answerText     # The first feasible answer is the best
            else:
                answerText = ""

            if answerText in bestAnswers:   continue
            bestAnswers[ answerText ] = (startIdx, endIdx, len(scores))
            scores += [ answerScore ]
            if len(scores) >= numTopLogits: break

        # To get the probability of "No-Answer", we need to always include it in the
        # best answers.
        if "" not in bestAnswers:
            bestAnswers[ "" ] = (0, 0, len(scores))
            scores += [ minNoAnswerScore ]

        if len(scores) == 0:
            myPrint("Warning: The \"bestAnswers\" is empty. No good answers for question \"%s\"")%(questionIdx)
            questionResults[questionIdx] = ("No-Answer!", 0, 0)
            return

        if doPrint: print("bestAnswers:\n", bestAnswers)
        
        # Apply softmax to get probabilities of the best answers:
        maxScores = max(scores)
        expScores = np.exp(np.array(scores) - maxScores)
        probs = expScores/expScores.sum()

        bestAnswerProb = probs[bestAnswers[ bestAnswer ][2]]
        noAnswerProb = probs[bestAnswers[ "" ][2]]
        if self.version >= 2:
            bestAnswerScore = scores(bestAnswers[ answerText ][2])
            scoreDiff = minNoAnswerScore - bestAnswerScore
            if scoreDiff > noAnswerScoreDiffThreshold:
                bestAnswer = ""
                
        if doPrint: print("bestAnswer:", bestAnswer)
        if doPrint: print("Prob(bestAnswer):", bestAnswerProb)
        questionResults[questionIdx] = (bestAnswer, bestAnswerProb, noAnswerProb)
