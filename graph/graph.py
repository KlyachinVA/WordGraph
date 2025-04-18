import scipy.sparse
import numpy as np

class Graph:
    def __init__(self,V,E):
        self.V = V
        self.E = E
        self.W = []
        self.neighbours()
        self.N = len(V)

    def AdjMatrix(self):
        self.Adj = np.zeros((self.N,self.N), dtype = int)

        for e in self.E:
            k = e[0]
            l = e[1]
            self.Adj[k,l] = 1
            self.Adj[l,k] = 1
        return self.Adj


    def put_weights(self,W):
        self.W = W

    def neighbours(self):
        nb = {}
        for e in self.E:
            if e[0] in nb:
                nb[e[0]].append(e[1])
                if e[1] in nb:
                    nb[e[1]].append(e[0])
                else:
                    nb[e[1]] = [e[0]]
            elif e[1] in nb:
                nb[e[1]].append(e[0])
                if e[0] in nb:
                    nb[e[0]].append(e[1])
                else:
                    nb[e[0]] = [e[1]]
            else:
                nb[e[0]] = [e[1]]
                nb[e[1]] = [e[0]]
        self.nb = nb
        return nb


    def round_graph(self):
        visited = []
        stack = []
        v = self.V[0]
        stack.append(v)

        while(len(stack)>0):
            v = stack.pop()
            visited.append(v)
            nbs = self.nb[v]
            for w in nbs:
                if w not in visited and w not in stack:
                    stack.append(w)
        return visited


    def make_chains(self):
        chains = []
        visited = []
        cur_chain = []
        stack = []
        v = self.V[0]
        stack.append(v)

        while(len(stack)>0):

            v = stack.pop()
            if v in visited:
                continue
            # print(self.nb)
            nbs = self.nb[v]
            # print(f"n[{v}] = {nbs}")
            # print(f"visited = {visited}")
            # print(f"stack = {stack}\n")
            if len(cur_chain)>0:
                if v not in self.nb[cur_chain[-1]]:

                    chains.append(cur_chain)
                    cur_chain = [v]

                else:
                    cur_chain.append(v)
            else:
                cur_chain.append(v)

            visited.append(v)

            for w in nbs:

                if w not in visited: #and w not in stack:
                    stack.append(w)
        chains.append(cur_chain)
        return chains



    def WienerIndex(self,factor=False):

        L = scipy.sparse.csgraph.shortest_path(self.W, method='auto', directed=False, return_predecessors=False)
        # L = L/((self.N -1)**2)
        # print(f"Inner distances ={L[0]}")
        if factor :
            return L.sum()/((self.N -1)**2)
        else:
            return L.sum()

    def InnerMetric(self):
        L = scipy.sparse.csgraph.shortest_path(self.W, method='auto', directed=True, return_predecessors=False)
        self.D = L
        return L

    def MeanDists(self):
        return np.sum(self.D,axis=1)





