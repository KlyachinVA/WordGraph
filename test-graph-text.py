from graph.graphtext import TextGraph
from graph.drawgraph import DrawGraph, DrawEuGraph
from graph.eutree import EuTree
from embedding import PreTrainedEmbeddings
import os
import matplotlib.pyplot as plt
import numpy as np

def test():
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/model.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-model-csv.bin")

    txt = "Раз два. Три четыре. Раз пять! Два семь, раз два три восемь."
    fname = "./data/texts/0/text1059_noninform.txt"
    f = open(fname)
    txt = f.read()
    gr = TextGraph(txt)
    # print(gr.sentences)
    # print(gr.E)
    # print(gr.WW)
    # DrawGraph(gr)
    P = gr.Means(embeddings)
    # print(P)
    print(P.shape)
    T = EuTree.CalcSpanTree(P,0)
    #DrawEuGraph(T)
    win = T.WienerIndex(True)
    winn = T.NormalizedWienerIndex(True)
    print(win,winn)


def calc_wiener_index_from_file(fname,emb):
    f = open(fname)
    txt = f.read()
    f.close()
    gr = TextGraph(txt)


    P = gr.Means(emb)
    # print(P.shape)
    T = EuTree.CalcSpanTree(P, 0)

    win = T.WienerIndex(True)
    winn = T.NormalizedWienerIndex(True)


    return win,winn

def calc_winer_index_from_path(path,emb):
    wins = []
    for fname in os.listdir(path):
        print(fname)
        win = calc_wiener_index_from_file(path + fname,emb)
        wins.append(win)
    wins = np.array(wins)
    return wins

def experiment_wiener_from_path():
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-tayga-csv.bin")

    path0 = "./data/texts/small-part/0/"
    path1 = "./data/texts/small-part/1/"
    wins0 = calc_winer_index_from_path(path0,embeddings)
    wins1 = calc_winer_index_from_path(path1,embeddings)
    n0 = len(wins0)
    n1 = len(wins1)
    X = np.zeros((n0+n1,2),dtype=float)
    Y = np.zeros((n0 + n1,),dtype=float)
    X[:n0,:] = wins0
    X[n0:n0+n1,:] = wins1
    Y[:n0] = 0
    Y[n0:n0+n1] = 1
    np.savez_compressed("./results/inform-dataset-part-300.npz",X=X,Y=Y)
    plt.plot(wins0[:,0],wins0[:,1],'.')
    plt.plot(wins1[:,0],wins1[:,1],'.')
    plt.show()

if __name__ == "__main__":
    # test()
    experiment_wiener_from_path()