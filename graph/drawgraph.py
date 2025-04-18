import matplotlib.pyplot as plt
import numpy as np
from .graph import Graph
from .eugraph import EuGraph

def DrawGraph(gr:Graph,color=(0,0,0)):
    N = len(gr.V)
    P = np.random.random((N,2))
    plt.plot(P[:,0],P[:,1],'o',color=color)
    delta = 0.01
    for i in range(N):
        plt.text(P[i,0]+delta,P[i,1]-delta,str(i))

    for e in gr.E:
        plt.plot([P[e[0],0],P[e[1],0]], [P[e[0],1],P[e[1],1]],color=color)
    plt.show()

def DrawEuGraph(gr:EuGraph,delta=0.01, color=(0,0,0),labels=[]):
    N = len(gr.V)
    P = gr.P

    plt.plot(P[:,0],P[:,1],'o',color=color)
    # delta = 0.01
    for i in range(N):
        if len(labels) == 0:
            plt.text(P[i,0]+delta,P[i,1]-delta,str(i))
        else:
            plt.text(P[i, 0] + delta, P[i, 1] - delta, str(i) + ": " + str(labels[i]))

    for e in gr.E:
        plt.plot([P[e[0],0],P[e[1],0]], [P[e[0],1],P[e[1],1]],color=color)
    plt.show()


