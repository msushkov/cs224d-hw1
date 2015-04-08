#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None):
        if not path:
            path = "cs224d/datasets/stanfordSentimentTreebank"

        self.path = path

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens
    
    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower().decode("utf-8").encode('latin1') for w in splitted]]
                
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)
        self._allsentences = [w for s in sentences for w in s]
        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        self.sentences()
        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        r = random.randint(C, len(allsent) - C - 1)
        context = allsent[r-C:r+C+1]
        return context

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in xrange(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            sent_labels[i] = labels[dictionary[full_sent]]
            
        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in xrange(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def samplingFreq(self):
        if hasattr(self, '_samplingFreq') and self._samplingFreq is not None:
            return self._samplingFreq

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        for i in xrange(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w] / self._wordcount
            # Reweigh
            freq = np.sqrt(min(1e-5, freq) * freq)
            samplingFreq[i] = freq

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq)

        self._samplingFreq = samplingFreq
        return self._samplingFreq

    def sampleTokenIdx(self):
        r = random.random()
        # binary search
        left = 0
        right = len(self.tokens()) - 1
        idx = (left + right) / 2
        freq = self.samplingFreq()
        while left < right:
            if freq[idx] > r:
                right = idx
                idx = (left + right) / 2
            elif freq[idx] < r:
                left = idx + 1
                idx = (left + right) / 2
            elif freq[idx] == r:
                return idx

        return left