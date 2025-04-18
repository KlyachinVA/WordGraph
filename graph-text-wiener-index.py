from graph.graphtext import TextGraph
from scipy.sparse.csgraph import connected_components
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from graph.eutree import EuTree


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
    txt = "Первое слово. Второе слово. Третье слово. Одни второе слова."
    fname = "./data/texts/0/text100_noninform.txt"
    f = open(fname)
    txt = f.read()
    grtxt = TextGraph(txt)
    print(grtxt.words)
    print(grtxt.WW)
    grtxt.put_weights(grtxt.WW)
    grtxt = connect_graph(grtxt)
    win = grtxt.WienerIndex(True)
    print(win)

def calc_wiener_index_from_file(fname):
    f = open(fname)
    txt = f.read()
    f.close()
    grtxt = TextGraph(txt)

    grtxt.put_weights(grtxt.WW)
    grtxt = connect_graph(grtxt)
    etr = EuTree.CalcSpanTreeOfEuGraph(grtxt,0)
    win = etr.WienerIndex(True)

    return win

def calc_winer_index_from_path(path):
    wins = []
    for fname in os.listdir(path):
        win = calc_wiener_index_from_file(path + fname)
        wins.append(win)
    wins = np.array(wins)
    return wins

def experiment_wiener_from_path():
    path0 = "./data/texts/00/"
    path1 = "./data/texts/11/"
    wins0 = calc_winer_index_from_path(path0)
    wins1 = calc_winer_index_from_path(path1)
    plt.plot(wins0,'.')
    plt.plot(wins1,'.')
    plt.show()

if __name__ == "__main__":
    # experiment()
    experiment_wiener_from_path()