from .tree import Tree
from .eugraph import EuGraph
import numpy as np
import math
MAXDIST = 1000000
class EuTree(EuGraph,Tree):

    def __init__(self,P,E):
        self.P = P
        self.N = len(P)
        self.n = len(P[0])
        V = list(range(self.N))
        self.V = V
        self.E = E

        self.put_eu_weights()


    def NearestPoint(self,p,Q,U):
        imin = 0
        dmin = ((p - Q[0])*(p - Q[0])).sum()
        first = True
        for i,q in enumerate(Q):
            d = ((p - q)*(p - q)).sum()
            if i not in U:
                if first:
                    dmin = d
                    imin = i
                    first = False
                else:
                    if d < dmin:
                        dmin = d
                        imin = i

        return imin,math.sqrt(dmin)

    def NearestPointOfEuGraph(self,k,Q,U,gr:EuGraph):
        imin = 0
        p = Q[U[k]]
        dmin = MAXDIST #((p - Q[0])*(p - Q[0])).sum()
        first = True
        for i in gr.nb[U[k]]:
            q = Q[i]
            d = ((p - q)*(p - q)).sum()
            if i not in U:
                if first:
                    dmin = d
                    imin = i
                    first = False
                else:
                    if d < dmin:
                        dmin = d
                        imin = i
        # print(f"Found e = {[k,imin]}, dist={dmin}")
        return imin,math.sqrt(dmin)


    def CalcG(self,Q,U):
        self.CalcEuDists()
        self.put_eu_weights()
        self.InnerMetric()
        D = self.MeanDists()
        G = np.zeros(self.N,dtype=np.float32)
        NP = np.zeros(self.N,dtype=int)

        for i,p in enumerate(self.P):
            k,d = self.NearestPoint(p,Q,U)
            G[i] = self.N * d + D[i]
            NP[i] = k

        return G, NP

    def CalcGOfEuGraph(self,Q,U,gr):
        self.CalcEuDists()
        self.put_eu_weights()
        self.InnerMetric()
        D = self.MeanDists()
        G = np.zeros(self.N,dtype=np.float32)
        NP = np.zeros(self.N,dtype=int)

        for i,p in enumerate(self.P):
            k,d = self.NearestPointOfEuGraph(i,Q,U, gr)
            G[i] = self.N * d + D[i]
            NP[i] = k

        return G, NP

    @staticmethod
    def CalcSpanTree(P,k):
        N = len(P)

        E = []

        U = [k]
        ind = 0
        #V = {k:ind}
        Ek = []
        while len(U) < len(P):
            Pk = np.array([P[j] for j in U])
            etr = EuTree(Pk, Ek)
            G,NP = etr.CalcG(P,U)
            # print(f"G,NP= {G}, {NP}")
            j = np.argmin(G)
            E.append([U[j], NP[j]])

            # print(f"U = {U}, E = {E}")
            U.append(NP[j])
            ind += 1
            #V[NP[j]] = ind
            # print(f"j={j}")
            #Ek.append([j,V[NP[j]]])
            Ek.append([j, ind])



        T = EuTree(P,E)
        return T

    @staticmethod
    def CalcSpanTreeOfEuGraph(gr:EuGraph, k):
        P = gr.P
        N = len(P)

        E = []

        U = [k]
        ind = 0
        # V = {k:ind}
        Ek = []
        while len(U) < len(P):
            Pk = np.array([P[j] for j in U])
            etr = EuTree(Pk, Ek)
            G, NP = etr.CalcGOfEuGraph(P, U,gr)
            # print(f"G,NP= {G}, {NP}")
            j = np.argmin(G)
            E.append([U[j], NP[j]])
            # print(f"Add edge {[U[j], NP[j]]}")
            # print(f"U = {U}, E = {E}")
            U.append(NP[j])
            ind += 1
            # V[NP[j]] = ind
            # print(f"j={j}")
            # Ek.append([j,V[NP[j]]])
            Ek.append([j, ind])

        T = EuTree(P, E)
        return T










