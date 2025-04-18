from graph.tree import Tree
from graph.graphtext import TextGraph
from graph.drawgraph import DrawGraph
from scipy.sparse.csgraph import connected_components
import random
import os
import numpy as np
import matplotlib.pyplot as plt

def connect_graph(gr):
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
    txt = "Раз два. Три четыре. Раз пять! Два семь, раз два три восемь."
    fname = "./data/texts/0/text100_noninform.txt"
    f = open(fname)
    txt = f.read()
    gr = TextGraph(txt)
    k = 0
    tr = Tree.CalcSpanTreeMinWiener(gr,gr.WW,k)
    DrawGraph(tr)
    tr = connect_graph(tr)
    win = tr.WienerIndex(True)
    print(win)

def calc_winer_index_from_file(fname):

    f = open(fname)
    txt = f.read()
    f.close()
    gr = TextGraph(txt)
    n = gr.N
    k = random.randint(0,n-1)
    tr = Tree.CalcSpanTreeMinWiener(gr, gr.WW, k)
    # DrawGraph(tr)
    tr = connect_graph(tr)
    win = tr.WienerIndex(True)
    return win

def calc_wiener_index_from_path(path):
    wins = []
    for fname in os.listdir(path):
        print(fname)
        try:
            win = calc_winer_index_from_file(path + fname)
            wins.append(win)
        except:
            pass
    return np.array(wins)

def experiment_wiener_from_paths():
    path0 = "./data/texts/part/0/"
    path1 = "./data/texts/part/1/"
    wins0 = calc_wiener_index_from_path(path0)
    wins1 = calc_wiener_index_from_path(path1)
    thres = 3.9
    # thres = 4.2
    n0 = len(wins0)
    nt0 = (wins0 > thres).sum()

    n1 = len(wins1)
    nt1 = (wins1 <= thres).sum()
    # print(wins0)
    # print(wins1)
    print(f"Accuracy = {100*(nt0 + nt1)/(n0 + n1)}%")
    plt.plot(wins0,'o')
    plt.plot(wins1,'o')
    plt.show()

if __name__ == "__main__":
    # experiment()
    experiment_wiener_from_paths()