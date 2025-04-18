from .graph import Graph
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
MAXDIST = 1000000
class Tree(Graph):

    def __init__(self,V,E):
        self.V = V
        self.E = E
        self.W = []
        self.N = len(V)
    def put_weights(self,W):
        self.W = W

    def to_parents(self,root):
        self.neighbours()
        seq = [-1 for i in range(len(self.V))]
        stack = []
        stack.append(root)
        while len(stack) > 0:
            parent = stack.pop()
            nb = self.nb[parent]
            for q in nb:
                if q != seq[parent]:
                    seq[q] = parent
                    stack.append(q)
        return seq




    def from_parents(self,seq):
        n = len(seq)
        self.V = [i for i in range(n)]
        root = 0
        for i,s in enumerate(seq):
            if s == -1:
                root = i
        E = []
        for i,s in enumerate(seq):
            if i != root:
                E.append([i,seq[i]])
        self.E = E

    def as_parents(self,root):
        seq = self.to_parents(root)
        parents={}

        for i,s in enumerate(seq):
            if i == root:
                continue
            if s in parents:
                parents[s].append(i)
            else:
                parents[s] = [i]
        return parents


    def weighted_center(self):
        nums = [-1 for i in range(len(self.E))]

        #pointed = []
        n = len(self.V)
        not_passed = [i for i in range(n)]
        nb = self.neighbours()
        num_not_pointed = {}
        not_pointed = {}
        pointed = {}
        res = -1
        for p in nb:
            pointed[p] = []



        for p in nb:
            num_not_pointed[p] = len(nb[p])
            not_pointed[p] = nb[p].copy()

        for i in range(n-1):
            unit_edges = [j for j in num_not_pointed if num_not_pointed[j]==1]
            print(f"листовые узлы: {unit_edges}")
            p_min = unit_edges[0]
            s_min = sum(pointed[p_min])
            k_min = not_pointed[p_min][0]
            # w_min = self.W[p_min][k_min] * (s_min + 1)
            w_min = (s_min + 1)

            for p in unit_edges:
                s = sum(pointed[p])

                k = not_pointed[p][0]
                #w = self.W[p][k] * (s + 1)
                w = (s + 1)
                print(f"s = {s}, p = {p}, w = {w}")
                if w < w_min:
                    w_min = w
                    p_min = p
                    s_min = s
                    k_min = k
                    res = k_min
            print(f"k_min = {k_min}, p_min = {p_min}, s_min = {s_min}")

            pointed[p_min].append(s_min + 1)
            pointed[k_min].append(s_min + 1)
            num_not_pointed[p_min] -= 1
            num_not_pointed[k_min] -= 1
            not_pointed[p_min].remove(k_min)
            not_pointed[k_min].remove(p_min)
            not_passed.remove(p_min)
            print(pointed)
        return res

    def NearestPointOfGraph(self,k,Q,U,gr:Graph):
        imin = 0
        p = U[k]
        dmin = MAXDIST #((p - Q[0])*(p - Q[0])).sum()
        first = True

        for i in gr.nb[p]:
            q = i
            d = gr.W[p][q]
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
        return imin,dmin

    def CalcGOfGraph(self,Q,U,gr):

        self.InnerMetric()
        D = self.MeanDists()
        G = np.zeros(self.N,dtype=np.float32)
        NP = np.zeros(self.N,dtype=int)

        for i,p in enumerate(self.V):
            k,d = self.NearestPointOfGraph(i,Q,U, gr)
            G[i] = self.N * d + D[i]
            NP[i] = k

        return G, NP


    @staticmethod
    def CalcSpanTreeMinWiener(gr:Graph,W,k):
        gr.put_weights(W)

        N = len(gr.V)

        E = []

        U = [k]
        ind = 0
        # V = {k:ind}
        Ek = []
        while len(U) < N:
            VTk = list(range(len(U)))
            tr = Tree(VTk, Ek)
            Wk = np.zeros((len(U),len(U)),dtype=float)
            for e in Ek:
                i0 = e[0]
                i1 = e[1]
                Wk[i0,i1] = W[U[i0],U[i1]]
                Wk[i1, i0] = W[U[i1], U[i0]]
            tr.put_weights(Wk)
            G, NP = tr.CalcGOfGraph(gr.V, U, gr)
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

        T = Tree(gr.V, E)
        WT = np.zeros((gr.N,gr.N),dtype=float)
        for e in E:
            i0 = e[0]
            i1 = e[1]
            WT[i0,i1] = gr.W[i0,i1]
            WT[i1, i0] = gr.W[i1, i0]
        T.put_weights(WT)

        return T

    @staticmethod
    def MST(W):
        # print(len(W))
        tcsr = minimum_spanning_tree(W)
        WT = tcsr.toarray()
        N = len(W)
        print(N)
        V = list(range(N))
        E = []
        for i in range(N):
            for j in range(i+1,N):
                if WT[i,j] > 0:
                    E.append([i,j])

        tr = Tree(V,E)
        tr.put_weights(WT)
        return tr













