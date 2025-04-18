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



def test1():
	N = 7
	P = np.random.rand(N,2)
	E = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[1,6],[2,6],[3,6],[4,6],[5,6]]
	egr = EuGraph(P,E)
	print(egr.W)
	DrawEuGraph(egr)
	T = EuTree.CalcSpanTreeOfEuGraph(egr,0)
	DrawEuGraph(T)
	
def test2():
    emb = PreTrainedEmbeddings.from_embeddings_file('./data/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-tayga-csv.bin")
  
    fname = "./data/texts/00/text101_noninform.txt"
    f = open(fname)
    txt = f.read()
    f.close()
	
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

    
    num_comp,labels = connected_components(egr.W,False,return_labels=True)
    # print(f"Num graphtext components = {num_comp}, labels = {labels}")
    DrawEuGraph(egr,labels=grtxt.norm_words)
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
    print(f"Num new graphtext components = {num_comp}, new labels = {labels}")
    T = EuTree.CalcSpanTreeOfEuGraph(egr, indx)
    # DrawEuGraph(T,labels=grtxt.norm_words)

    num_comp, labels = connected_components(T.W, False, return_labels=True)
    print(f"Num  components in tree = {num_comp},  labels for tree = {labels}")
    DrawEuGraph(T,labels=words)
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

	
if __name__ == "__main__":
	#test1()
	test2()
	


