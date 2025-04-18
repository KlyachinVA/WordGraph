from .graph import Graph

import numpy as np

class EuGraph(Graph):

    def __init__(self,P,E):
        self.N = len(P)
        self.n = len(P[0])
        self.P = P
        self.E = E
        V = list(range(self.N))
        super().__init__(V,E)
        self.put_eu_weights()

    def CalcEuDists(self):
        D = np.zeros((self.N,self.N),dtype=np.float32)

        for k in range(self.N):
            Q = self.P - self.P[k]
            D[k] = np.sqrt((Q*Q).sum(axis=1))

        self.EuD = D

    def put_eu_weights(self):
        self.CalcEuDists()
        self.AdjMatrix()
        W = self.EuD * self.Adj
        self.put_weights(W)

    def NormalizedWienerIndex(self,factor=False):
        IW = self.WienerIndex(factor)
        EUIW = self.EuD.sum()
        return IW/EUIW







