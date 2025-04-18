# from pretrained_embeddings import PreTrainedEmbeddings
from embedding import PreTrainedEmbeddings
import pymorphy2 as pm2
import matplotlib.pyplot as plt
from graph.eutree import EuTree
import random
import numpy as np
import os
from graph.drawgraph import DrawGraph, DrawEuGraph
from graph.graphtext import TextGraph
from graph.eugraph import EuGraph
def test_to_vector():
    # txt = "Во глубине сибирских руд, храните гордое терпенье - не пропадет\
    # ваш скорбный труд и дум высокое стремленье"

    # txt = "сохнут дело парта книга пилить река смотреть пить листок трезвый"
    # txt = "Под собою ног не чую и качается земля третий месяц я бичую так как списан подчистую с\
    #        китобоя корабля"
    # txt = "У лукоморья дуб зелёный златая цепь на дубе том и днём и ночью кот учёный все ходит по цепи кругом"

    # txt = "дифференцируемая функция является непрерывной в каждой точке отрезка"
    # txt = "хорошо живёт на свете вини-пух от того поёт он эти песни вслух"
    # txt = "книга слесарь молоток гармонь огород решать вода светить ваза потолок"
    # txt = "рука говорить делать видеть жизнь самый глаз смотреть очень хотеть понимать становиться день давать друг дело"
    # txt = "Нужно найти хорошую компанию пройти не одно собеседование затем возможно ещё и проверку службы безопасности Чтобы было легче справиться с этими задачами собрали в помощь три статьи"
    # txt = "стол небо говорить принять договор идти"
    # txt = "нужный оставаться думать сторона каждый что-то дом брать"
    fname = "./data/texts/text-17.txt"
    f = open(fname)
    txt = f.read()
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/model.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-model-csv.bin")
    vectors = []
    words = []
    for word in txt.split():

        norm_word,v = embeddings.get_word_vector(word)
        try:
            if v.all() != None and norm_word not in words:
                vectors.append(v)
                words.append(norm_word)
        except:
            pass

    vectors = np.array(vectors)
    print(vectors.shape)

    N = len(vectors)
    ind = random.randint(0,N - 1)
    T = EuTree.CalcSpanTree(vectors, ind)
    print(f"TE = {T.E}")
    print(f"Index Wiener = {T.WienerIndex(True)}, ind = {ind}")
    # print(f"EuDists = {T.EuD}")
    DrawEuGraph(T, delta = 0.002, color=(1, 0, 0),labels=words)

def calc_wiener_index_graph(txt,emb):
    grtxt = TextGraph(txt)

    vectors = []
    words = grtxt.words
    norm_words = []
    for word in words:
        norm_word, v = emb.get_word_vector(word)
        try:
            if v.all() != None:

                vectors.append(v)
                norm_words.append(norm_word)
        except:
            v = np.zeros((300,),dtype=float)
            vectors.append(v)
            norm_words.append(norm_word)


    vectors = np.array(vectors)
    # print(vectors.shape)

    N = len(vectors)
    ind = random.randint(0, N - 1)
    egr = EuGraph(vectors,grtxt.E)
    T = EuTree.CalcSpanTreeOfEuGraph(egr, ind)
    wind0 = T.WienerIndex(True)
    wind1 = T.NormalizedWienerIndex()
    # print(f"TE = {T.E}")
    print(f"Index Wiener = {wind0}, NIM = {wind1}, ind = {ind}")

    return [wind0,wind1]

def calc_wiener_index(txt,emb):
    vectors = []
    words = []
    for word in txt.split():

        norm_word, v = emb.get_word_vector(word)
        try:
            if v.all() != None and norm_word not in words:
                vectors.append(v)
                words.append(norm_word)
        except:
            pass

    vectors = np.array(vectors)
    # print(vectors.shape)

    N = len(vectors)
    ind = random.randint(0, N - 1)
    T = EuTree.CalcSpanTree(vectors, ind)
    wind0 = T.WienerIndex(True)
    wind1 = T.NormalizedWienerIndex()
    # print(f"TE = {T.E}")
    print(f"Index Wiener = {wind0}, NIM = {wind1}, ind = {ind}")

    return [wind0,wind1]
def calc_index_wiener_from_path(path):
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/tayga_1_2.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-tayga-csv.bin")
    winds = []
    for fname in os.listdir(path):
        f = open(path + fname)
        txt = f.read()
        wind = calc_wiener_index(txt,embeddings)
        winds.append(wind)
        f.close()
    return np.array(winds)

def calc_index_wiener_graph_from_path(path):
    embeddings = PreTrainedEmbeddings.from_embeddings_file('./data/model.csv',
                                                           build_index=False,
                                                           index_file="./indexes/index-model-csv.bin")
    winds = []
    for fname in os.listdir(path):
        f = open(path + fname)
        txt = f.read()
        wind = calc_wiener_index_graph(txt,embeddings)
        winds.append(wind)
        f.close()
    return np.array(winds)

def experiment():
    path0 = "./data/texts/0/"
    path1 = "./data/texts/1/"

    w0 = calc_index_wiener_from_path(path0)
    w1 = calc_index_wiener_from_path(path1)
    plt.plot(w0[:,0],w0[:,1],'.')
    plt.plot(w1[:,0],w1[:,1], '.')
    plt.show()

def experiment_graph():
    path0 = "./data/texts/0/"
    path1 = "./data/texts/1/"

    w0 = calc_index_wiener_graph_from_path(path0)
    w1 = calc_index_wiener_graph_from_path(path1)
    plt.plot(w0[:,0],w0[:,1],'.')
    plt.plot(w1[:,0],w1[:,1], '.')
    plt.show()
if __name__ == "__main__":
    # test_to_vector()
    # experiment()
    experiment_graph()