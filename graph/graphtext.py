from .graph import Graph
import re
import pymorphy2 as pm2
import numpy as np
from nltk.corpus import stopwords
morph = pm2.MorphAnalyzer()

class TextGraph(Graph):

    def __init__(self,txt,thres=0):
        sw = stopwords.words('russian')
        # print(sw)
        rg = re.compile(r"[.!?]")
        rx = re.compile(r"[^а-яёА-яЁ ]")
        rxdefis = re.compile(r"[-:]")
        sents = rg.split(txt)
        sentences = []
        words = []
        for sent in sents:
            sent = rxdefis.sub(" ", sent)
            sent = rx.sub("",sent)

            ws = sent.split()
            wsn = []
            for w in ws:
                res = morph.parse(w)[0]
                res = res.normalized
                if res.word not in sw:
                    # print(res.word)
                    wsn.append(res.word)

            words += wsn
            sentences.append(wsn)

        words = list(set(words))

        N = len(words)
        # print(N, words)
        V = [i for i in range(N)]
        E = []
        word_to_index = {}
        word_frequency = {}
        self.words = words

        for k,word in enumerate(words):
            word_to_index[word] = k
        self.sentences = sentences
        for sent in self.sentences:
            # print(sent)
            for word in sent:
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1


        self.word_frequency = word_frequency
        self.word_to_index = word_to_index

        # print("F=",self.word_frequency)
        # print(word_to_index)
        # self.clean_words(thres)
        self.WW = np.zeros((N,N),dtype=float)
        for sent in self.sentences:
            L = len(sent)
            for i in range(L):
                for j in range(i+1,L):
                    w1 = sent[i]
                    w2 = sent[j]
                    k1 = word_to_index[w1]
                    k2 = word_to_index[w2]
                    E.append([k1,k2])
                    self.WW[k1,k2] = 1
                    self.WW[k2, k1] = 1


        super().__init__(V,E)

    def clean_words(self,thres):

        new_sents = []

        for sent in self.sentences:
            new_sent = []
            for word in sent:
                if self.word_frequency[word] > thres:
                    new_sent.append(word)
            new_sents.append(new_sent)
        self.sentences = new_sents


    def Means(self,emb):

        P = []
        for sent in self.sentences:
            if len(sent) == 0:
                continue
            mean = []
            for word in sent:
               nword, vec = emb.get_word_vector(word)
               # print(nword,len(vec))

               try:
                   if vec.all() != None:
                       mean.append(vec)
               except:
                   pass
            mean = np.array(mean)
            mvec = mean.mean(axis=0)
            P.append(mvec)
        return np.array(P)














