from graph.graphtext import TextGraph
from graph.tree import Tree
from scipy.sparse.csgraph import connected_components
import numpy as np
import random
def connect_graph(gr):
    W = gr.WW

    W = np.array(W)
    gr.put_weights(W)
    num_comp, labels = connected_components(gr.W, False, return_labels=True)
    # print(f"Num components = {num_comp}, labels = {labels}")

    verts_by_comps = {}
    for i, lb in enumerate(labels):
        if lb in verts_by_comps:
            verts_by_comps[lb].append(i)
        else:
            verts_by_comps[lb] = [i]
    ends = []
    # print(f"verts_by_comps = {verts_by_comps}")
    for comp in verts_by_comps:
        l_comp = len(verts_by_comps[comp])
        ind = random.randint(0, l_comp - 1)
        ends.append(verts_by_comps[comp][ind])

    # print(f"New ends = {ends}")
    for j in range(len(ends) - 1):
        i1 = ends[j]
        i2 = ends[j + 1]
        gr.E.append([i1, i2])
        # p1 = egr.P[i1]
        # p2 = egr.P[i2]

        gr.W[i1, i2] = 1.0
        gr.W[i2, i1] = 1.0
    return gr


def experiment():
    fname = "./data/texts/00/text100_noninform.txt"
    f = open(fname)
    txt = f.read()
    f.close()
    gr =TextGraph(txt)
    gr = connect_graph(gr)
    mst = Tree.MST(gr.W)
    print(mst.E)

if __name__ == "__main__":
    experiment()