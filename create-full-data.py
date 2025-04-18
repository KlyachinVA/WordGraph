from graph.tree import Tree
from graph.eugraph import EuGraph
from scipy.sparse.csgraph import connected_components
from graph.graphtext import TextGraph
from graph.drawgraph import DrawGraph, DrawEuGraph
from graph.eutree import EuTree
from embedding import PreTrainedEmbeddings
import os
import matplotlib.pyplot as plt
import numpy as np
import random


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

def calc_wiener_index_graph(txt,emb):
    grtxt = TextGraph(txt)

    vectors = []
    words = grtxt.words
    norm_words = []
    first = True
    for word in words:
        norm_word, v = emb.get_word_vector(word)
        try:
            if v.all() != None:

                vectors.append(v)
                norm_words.append(norm_word)
                # if first:
                #     # print(f"{word}: {v}")
                #     first=False
        except:
            print("ADD RANDOM")
            v = np.random.rand(300)
            vectors.append(v)
            norm_words.append(norm_word)
    # print(len(norm_words),norm_words)
    grtxt.norm_words = norm_words
    # print(f"norm_words={norm_words}")
    # print(f"words={words}")
    vectors = np.array(vectors)
    # print(vectors.shape)

    N = len(vectors)

    indx = random.randint(0, N - 1)
    print(f"N = {N}, ind = {indx}")
    egr = EuGraph(vectors,grtxt.E)

    # DrawEuGraph(egr,labels=grtxt.norm_words)
    num_comp,labels = connected_components(egr.W,False,return_labels=True)
    # print(f"Num graphtext components = {num_comp}, labels = {labels}")

    verts_by_comps = {}
    for i,lb in enumerate(labels):
        if lb in verts_by_comps:
            verts_by_comps[lb].append(i)
        else:
            verts_by_comps[lb] = [i]
    ends = []
    # print(f"verts_by_comps = {verts_by_comps}")
    for comp in verts_by_comps:
        l_comp = len(verts_by_comps[comp])
        ind = random.randint(0,l_comp - 1)
        ends.append(verts_by_comps[comp][ind])

    # print(f"New ends = {ends}")
    for j in range(len(ends) - 1):
        i1 = ends[j]
        i2 = ends[j+1]
        egr.E.append([i1,i2])
        # p1 = egr.P[i1]
        # p2 = egr.P[i2]
        d12 = egr.EuD[i1,i2]
        egr.W[i1,i2] = d12
        egr.W[i2,i1] = d12

    num_comp, labels = connected_components(egr.W, False, return_labels=True)
    # print(f"Num new graphtext components = {num_comp}, new labels = {labels}")
    T = EuTree.CalcSpanTreeOfEuGraph(egr, indx)
    # DrawEuGraph(T,labels=grtxt.norm_words)

    num_comp, labels = connected_components(T.W, False, return_labels=True)
    # print(f"Num  components in tree = {num_comp},  labels for tree = {labels}")

    verts_by_comps = {}
    for i, lb in enumerate(labels):
        # print(i,lb)
        if lb in verts_by_comps:
            verts_by_comps[lb].append(i)
        else:
            verts_by_comps[lb] = [i]
    ends = []
    # print(f"components={verts_by_comps}")
    for comp in verts_by_comps:
        l_comp = len(verts_by_comps[comp])
        ind = random.randint(0, l_comp - 1)
        ends.append(verts_by_comps[comp][ind])
    # print(f"ends={ends}")
    for j in range(len(ends) - 1):
        i1 = ends[j]
        i2 = ends[j + 1]
        T.E.append([i1, i2])
        # p1 = egr.P[i1]
        # p2 = egr.P[i2]
        d12 = egr.EuD[i1, i2]

        # print(f"d12={d12}, word{i1} = {words[i1]}, word{i2} {words[i2]}")
        T.W[i1, i2] = d12
        T.W[i2, i1] = d12
    # connect_graph(T)
    # num_comp, labels = connected_components(T.W, False, return_labels=True)
    # print(f"Num new components in tree = {num_comp},  new labels for tree = {labels}")
    factor = True
    wind0 = T.WienerIndex(factor)
    wind1 = T.NormalizedWienerIndex(factor)
    # print(f"TE = {T.E}")
    print(f"Index Wiener = {wind0}, NIM = {wind1}, ind = {indx}")

    return [wind0,wind1]


def calc_wiener_index_from_file(fname,emb,thres=0):
    f = open(fname)
    txt = f.read()
    f.close()
    gr = TextGraph(txt,thres)

    W = gr.WW
    W = np.array(W)
    gr.put_weights(W)
    P = gr.Means(emb)
    # print(P.shape)
    M = len(P)
    k = random.randint(0,M-1)
    T = EuTree.CalcSpanTree(P, k)

    win0 = T.WienerIndex(True)
    win1 = T.NormalizedWienerIndex(True)
    print(win0,win1)
    n = gr.N
    k = random.randint(0, n - 1)
    tr = Tree.CalcSpanTreeMinWiener(gr, gr.WW, k)
    # DrawGraph(tr)
    tr = connect_graph(tr)
    win2 = tr.WienerIndex(True)
    print(win2)
    gr = connect_graph(gr)
    mst = Tree.MST(gr.W)
    # print(mst.E)
    win3 = mst.WienerIndex(True)
    print(win3)
    win4,win5 = calc_wiener_index_graph(txt,emb)
    return win0,win1,win2 ,win3 ,win4,win5

def calc_winer_index_from_path(path,emb,thres):
    wins = []
    for fname in os.listdir(path):
        print(fname)
        try:
            win = calc_wiener_index_from_file(path + fname,emb,thres)
            wins.append(win)
            print(win)
        except:
            pass
    wins = np.array(wins)
    return wins

def experiment_wiener_from_path():
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-tayga-csv.bin")
    thres = 0
    # path0 = "./data/texts/00/"
    # path1 = "./data/texts/11/"
    # path0 = "./data/texts/00/"
    # path1 = "./data/texts/11/"
    # path0 = "./data/texts/0/"
    # path1 = "./data/texts/1/"
    path0 = "./data/texts/small-part/0/"
    path1 = "./data/texts/small-part/1/"
    wins0 = calc_winer_index_from_path(path0,embeddings,thres)
    wins1 = calc_winer_index_from_path(path1,embeddings,thres)
    n0 = len(wins0)
    n1 = len(wins1)
    num = 6
    X = np.zeros((n0+n1,num),dtype=float)
    Y = np.zeros((n0 + n1,),dtype=float)
    X[:n0,:] = wins0
    X[n0:n0+n1,:] = wins1
    Y[:n0] = 0
    Y[n0:n0+n1] = 1
    np.savez_compressed("./results/inform-dataset-6.npz",X=X,Y=Y)
    plt.plot(wins0[:,0],wins0[:,1],'.')
    plt.plot(wins1[:,0],wins1[:,1],'.')
    plt.show()

def test_calc_wiener_index_graph_text():
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-tayga-csv.bin")
    fname = "./data/texts/000/text100_noninform.txt"
    f = open(fname)
    txt = f.read()
    w1,w2 = calc_wiener_index_graph(txt,embeddings)
    print(w1,w2)


if __name__ == "__main__":
    experiment_wiener_from_path()
    # test_calc_wiener_index_graph_text()