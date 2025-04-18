import numpy as np
from .graph import Graph


class Triangulation:

    def __init__(self,P,T):
        self.P = P
        self.T = T
        self.N = len(P)
        self.n = len(P[0])
        self.M = len(T)

    def create_graph(self):
        V = [i for i in range(len(self.T))]
        E = []
        S = {}

        for k,tr in enumerate(self.T):
            tr.sort()
            for i in range(len(tr)):
                e1 = tr[:i]
                e2 = tr[i+1:]
                e = tuple(e1 + e2)
                if e in S:
                    S[e].append(k)
                else:
                    S[e] = [k]


            for e in S:
                if len(S[e]) == 2:
                    E.append(S[e])

        gr = Graph(V,E)
        return gr


