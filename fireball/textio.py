# Copyright (c) 2020 InterDigital AI Research Lab
"""
This file contains the implementation for fireball's tokenization functionality for different NLP tasks.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    -----------------------------------------------------------------------------
# 08/10/2020    Shahab Hamidi-Rad       First version of the file.
# 08/12/2016    Shahab                  Added support for word-piece tokenization.
# **********************************************************************************************************************
import unicodedata
import numpy as np

# **********************************************************************************************************************
def isChinese(c):
    # This is taken from the original BERT implementation. See the "_is_chinese_char" function in:
    #       https://github.com/google-research/bert/blob/master/tokenization.py
    
    cCode = ord(c)
    if cCode >= 0x4E00 and cCode <= 0x9FFF:     return True
    if cCode >= 0x3400 and cCode <= 0x4DBF:     return True
    if cCode >= 0x20000 and cCode <= 0x2A6DF:   return True
    if cCode >= 0x2A700 and cCode <= 0x2B73F:   return True
    if cCode >= 0x2B740 and cCode <= 0x2B81F:   return True
    if cCode >= 0x2B820 and cCode <= 0x2CEAF:   return True
    if cCode >= 0xF900 and cCode <= 0xFAFF:     return True
    if cCode >= 0x2F800 and cCode <= 0x2FA1F:   return True
    return False

# **********************************************************************************************************************
def shouldSkip(c):
    cCode = ord(c)
    if cCode in [ 0, 0xfffd]:                   return True     # Special Cases
    if c in ["\t", "\n", "\r"]:                 return False    # Tab, New line, Carriage return
    if unicodedata.category(c) in ["Cc", "Cf"]: return True     # Control Chars
    return False

# **********************************************************************************************************************
def isWhiteSpace(c):
    if c in [" ", "\t", "\n", "\r"]:            return True
    if unicodedata.category(c) == "Zs":         return True
    return False

# **********************************************************************************************************************
def isPunctuation(c):
    cCode = ord(c)
    if cCode >= 33 and cCode <= 47:             return True
    if cCode >= 58 and cCode <= 64:             return True
    if cCode >= 91 and cCode <= 96:             return True
    if cCode >= 123 and cCode <= 126:           return True
    if unicodedata.category(c)[0] == "P":       return True
    return False

# **********************************************************************************************************************
class Tokenizer:
    # ******************************************************************************************************************
    def __init__(self, vocabInfo, uncased=True, reservedTokens=set()):
        if isinstance(vocabInfo, str):      vocabList = self.loadVocab(vocabInfo)
        elif isinstance(vocabInfo, list):   vocabList = vocabInfo
        else:
            raise ValueError("\"vocabInfo\" must be either a file name or a list of strings from the vocabulary!")

        self.token2Id = {}
        self.id2Token = vocabList
        for id, token in enumerate(self.id2Token):   self.token2Id[ token ] = id

        self.uncased = uncased
        # Note: Reserved Tokens should not be lower case
        self.reservedTokens = reservedTokens
        
        self.unknown = {'tok': "[UNK]", 'id': self.token2Id["[UNK]"]}
        self.cls     = {'tok': "[CLS]", 'id': self.token2Id["[CLS]"]}
        self.sep     = {'tok': "[SEP]", 'id': self.token2Id["[SEP]"]}
        self.pad     = {'tok': "[PAD]", 'id': self.token2Id["[PAD]"]}

    # ******************************************************************************************************************
    def loadVocab(self, vocabFileName):
        with open(vocabFileName, "r", encoding="utf-8") as vocabFile:
            tokens = vocabFile.readlines()
        return [token.strip() for token in tokens]

    # ******************************************************************************************************************
    def addTokenToVocab(self, token):
        if token in self.token2Id:  return
        self.token2Id[ token ] = len(self.id2Token)
        self.id2Token += [token]

    # ******************************************************************************************************************
    def addReservedTokens(self, tokens):
        if isinstance(tokens, str):
            self.reservedTokens.add(tokens)
            self.addTokenToVocab(tokens)
        else:
            for token in tokens:
                self.reservedTokens.add(token)
                self.addTokenToVocab(token)

    # ******************************************************************************************************************
    @property
    def vocabSize(self):
        return len(self.id2Token)

    # ******************************************************************************************************************
    def normalizeText(self, text):
        # If we are tokenizing "uncased", this function modifies a text by separating
        # the accents from the actual characters and then removing them while making sure
        # the length of text and the character indexes remains unchanged.
        # Note:
        # - We do not replace the un-accompanied accents in the text because it would change
        #   the length and indexes. We replace them with 'X' so that they don't get removed.
        # - Some asian characters may have more than one accents. This also can mess up the
        #   indexes. So we replace them with X also.
        
        if not self.uncased:    return text
        
        # To keep the indexes and length unchanged, keep the un-accompanied accents and
        # asian characters in the original text. Here we use "Lo" unicode category which
        # includes the asian characters.
        normText = ""
        for c in text:
            if unicodedata.category(c) in ["Mn", "Lo"]: normText += 'X'
            else:                                       normText += c.lower()

        normText = unicodedata.normalize("NFD", normText)
        
        # Remove the accents
        normText = "".join(c for c in normText if unicodedata.category(c) != "Mn")
                    
        assert len(normText) == len(text), "Norm(%d) != Org(%d)"%(len(normText) , len(text))
        return normText
    
    # ******************************************************************************************************************
    def tokenize(self, text):
        # First clean and split based on white spaces
        curToken, orphanActuals, curActuals = "", "", []
        rawTokens = []
        for c in text:
            if shouldSkip(c):
                orphanActuals += c
                continue
                
            elif isChinese(c):
                if curToken!="":
                    curActuals[-1] += orphanActuals
                    rawTokens += [(curToken, curActuals)]
                    rawTokens += [(c, [c])]
                else:
                    rawTokens += [(c, [orphanActuals+c])]
                curToken, orphanActuals, curActuals = "", "", []
                
            elif isWhiteSpace(c):
                if curToken != "":
                    curActuals[-1] += orphanActuals + c
                    rawTokens += [ (curToken, curActuals) ]
                    curToken, orphanActuals, curActuals = "", "", []
                else:
                    orphanActuals += c
            else:
                curToken += c
                curActuals += [ orphanActuals + c ]
                orphanActuals = ""
            
        if curToken != "":
            curActuals[-1] += orphanActuals
            rawTokens += [ (curToken, curActuals) ]
        elif orphanActuals != "":
            if len(rawTokens) == 0:     return [], []
            actuals = rawTokens[-1][1]
            actuals[-1] += orphanActuals
            rawTokens[-1] = (rawTokens[-1][0], actuals)
        
        # More cleanup: Accents and punctuation
        tokens = []
        orphanActuals = ""
        for token, actuals in rawTokens:
            assert len(token)==len(actuals), "'%s' -> %s"%(token, str(actuals))
            if token in self.reservedTokens:
                tokens += [ (token, actuals) ]
                continue
                
            curToken, curActuals = "", []
            for i, c in enumerate(token):
                if self.uncased:
                    normC = unicodedata.normalize("NFD", c.lower())
                    c = normC[0]
                    if len(normC)>1:
                        for nc in normC:
                            if unicodedata.category(nc) in ["Mn"]: continue
                            c = nc
                            break

                    if unicodedata.category(c) in ["Mn"]:
                        orphanActuals += actuals[i]
                        continue

                # Split each token further if it contains punctuations
                if isPunctuation(c):
                    if curToken!="":
                        curActuals[-1] += orphanActuals
                        tokens += [(curToken, curActuals)]
                        tokens += [(c, [actuals[i]])]
                    else:
                        actuals[i] = orphanActuals + actuals[i]
                        tokens += [(c, [actuals[i]])]
                    curToken, orphanActuals, curActuals = "", "", []

                else:
                    curToken += c
                    curActuals += [ orphanActuals + actuals[i] ]
                    orphanActuals = ""
               
            if curToken != "":
                curActuals[-1] += orphanActuals
                tokens += [ (curToken, curActuals) ]
                orphanActuals = ""
            elif orphanActuals != "":
                if len(tokens) > 0:
                    actuals = tokens[-1][1]
                    actuals[-1] += orphanActuals
                    tokens[-1] = (tokens[-1][0], actuals)
                    orphanActuals = ""

        # Split each token based on WordPiece Vocab
        wordpieceTokens = []
        orgTexts = []
        for token, actuals in tokens:
            assert len(token)==len(actuals), "'%s' -> %s"%(token, str(actuals))
            # Keep reserved tokens
            if token in self.reservedTokens:
                wordpieceTokens += [ (token, actuals) ]
                continue
            
            # Very large tokens are marked as unknown
            if len(token) > 100:
                wordpieceTokens += [ (self.unknown['tok'], actuals) ]
                continue
            
            start = 0
            subTokens = []
            subTokenOrgs = []
            while start < len(token):
                end = len(token)
                prefix = "##" if start else ""
                while start<end:
                    subToken = prefix + token[start:end]
                    subTokenOrg = "".join(actuals[start:end])
                    if subToken in self.token2Id:   break
                    subToken = None
                    end -= 1
                
                if subToken is None:
                    subTokens = [ self.unknown['tok'] ]
                    subTokenOrgs = [ "".join(actuals) ]
                    break
                
                subTokens += [ subToken  ]
                subTokenOrgs += [ subTokenOrg ]
                start = end
                
            wordpieceTokens += subTokens
            orgTexts += subTokenOrgs
        
        if len(wordpieceTokens) == 0:   return [], []
        
        assert len(wordpieceTokens) == len(orgTexts)
        
        # Covert orgTexts to spans
        spans = []
        start = 0
        for txt in orgTexts:
            spans += [ [start, start+len(txt)] ]
            start += len(txt)
        assert spans[-1][-1] == len(text)

        return wordpieceTokens, spans
    
    # ******************************************************************************************************************
    def join(self, tokens):
        return " ".join(tokens).replace(" ##", "").strip()

    # ******************************************************************************************************************
    def toIds(self, tokens):
        return [self.token2Id[t] for t in tokens]
        
    # ******************************************************************************************************************
    def encode(self, text, maxLen=None):
        tokens, spans = self.tokenize(text)
        if maxLen is not None:
            tokens = tokens[:maxLen]
            spans = spans[:maxLen]
        tokenIds = [ self.token2Id[token] if token in self.token2Id else self.unknown['id'] for token in tokens]
        return tokenIds, spans

    # ******************************************************************************************************************
    def decode(self, tokenIds):
        tokens = [self.id2Token[tokenId] for tokenId in tokenIds]
        return self.join(tokens)

    # ******************************************************************************************************************
    def packTokenIds(self, tokenIds1, tokenIds2, seqLen, packMethod='101_tok1_102_tok2_102_pad'):
        if packMethod == '101_tok1_102_tok2_102_pad':
            tokIds = [self.cls['id']] + tokenIds1 + [self.sep['id']]
            if tokenIds2 is not None:   tokIds += tokenIds2 + [self.sep['id']]
            noPadLen = len(tokIds)
            assert noPadLen<=seqLen, "%d+%d+%d=%d>%d"%(len(tokenIds1),
                                                       0 if tokenIds2 is None else len(tokenIds2),
                                                       2 if tokenIds2 is None else 3,
                                                       noPadLen, seqLen)
            tokIds += [ self.pad['id'] ]*(seqLen-noPadLen)
            
            typeIds = [0]*(len(tokenIds1)+2)
            if tokenIds2 is not None:   typeIds += [1]*(len(tokenIds2)+1)
            typeIds += [0]*(seqLen-noPadLen)
            
            return (tokIds, typeIds)
            
    # ******************************************************************************************************************
    def makeModelInput(self, context, question, maxQLen=None, seqLen=None, returnSpans=False):
        # This is used for inference. Note that when inferring one sample, there is no need for padding.
        # So, noPadLen in this case is equal to the length of the sequence (question+context+3)
        # The spans can be used to generate the answer that exactly matches the text from context. See
        # the getTextFromTokSpan function below.
        questionTokIds, questionSpans = self.encode(question)
        contextTokIds, contextSpans = self.encode(context)
        
        if maxQLen is not None:
            questionTokIds = questionTokIds[:maxQLen]
            questionSpans = questionSpans[:maxQLen]
            
        if seqLen is not None:
            maxContextLen = seqLen - len(questionTokIds)-3
            assert len(contextTokIds) <= maxContextLen, "Context too long! It needs to be sliced to several samples!"
            # TODO: implement segmentation here!

        tokIds = [self.cls['id']] + questionTokIds + [self.sep['id']] + contextTokIds + [self.sep['id']]
        typeIds = [0]*(len(questionTokIds)+2) + [1]*(len(contextTokIds)+1)
        
        if returnSpans:
            return (tokIds, typeIds), (contextSpans, questionSpans)
        
        return (tokIds, typeIds)
        
    # ******************************************************************************************************************
    def getTextFromTokSpan(self, seq, context, spans, startTokIdx, endTokIdx):
        # spans contains one (start,end) tuple for each token in context.
        contextOffset = seq.index(self.sep['id'])+1
        startIdx = spans[startTokIdx - contextOffset][0]
        endIdx = spans[endTokIdx - contextOffset][1]
        return context[ startIdx : endIdx ]

#tokenizer = Tokenizer("bert-base-uncased-vocab.txt")
#tokenizer.addReservedTokens(["[UNK]", "[SEP]"])
#tokens = tokenizer.tokenize("Sha[h(ab [SEP] Hamidi-Rad [UNK]")
#print(tokens)
#print( tokenizer.join(tokens))
#print( tokenizer.encode("Sha[h(ab [SEP] Hamidi-Rad [UNK]"))
