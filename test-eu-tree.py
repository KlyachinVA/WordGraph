from graph.eutree import EuTree
from graph.eugraph import EuGraph
import numpy as np
from graph.drawgraph import DrawGraph, DrawEuGraph

def test():
    P = np.array([[0,0],[1.0,0],[1.0,1.0],[0,1.0],[0.2,0.5],[0.8,0.5]])
    p = 0
    E = [[0,4],[3,4],[1,5],[2,5],[4,5]]
    etr = EuTree(P,E)
    etr.CalcEuDists()
    print(etr.EuD)
    etr.put_eu_weights()
    print(f"W = {etr.W}")
    print(f"IndWiener = {etr.WienerIndex()}")
    IM = etr.InnerMetric()
    print(IM)
    delta = etr.MeanDists()
    print(delta)
    Q = np.array([[0.25,0.25],[0.75,0.25]])
    G,NP = etr.CalcG(Q,[])
    print(f"G = {G}, NP = {NP}")

    T = EuTree.CalcSpanTree(P,0)
    print(f"TE = {T.E}")
    DrawEuGraph(T,color=(1,0,0))
    EU = [[0,4],[3,4],[1,5],[2,5],[4,5],[0,1],[1,2],[2,3],[3,0]]
    egr = EuGraph(P,EU)
    ET = EuTree.CalcSpanTreeOfEuGraph(egr,0)
    print(f"ET = {ET.E}")
    DrawEuGraph(ET,color=(0,0,1))


if __name__ == "__main__":
    test()