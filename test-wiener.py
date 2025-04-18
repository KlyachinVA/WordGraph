from graph import Graph
import numpy as np
from tree import Tree

def test():
    V = [0,1,2,3]
    E = [[0,1],[1,2],[2,3],[3,0]]
    E1 = [[0,1],[1,2],[2,3]]
    W = np.array([[0,1,0,1],
         [1,0,1,0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]
         ])
    W1 = np.array([[0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]
                  ])

    gr = Graph(V,E)
    gr.put_weights(W)
    res = gr.WienerIndex()
    print(res)

    tr = Tree(V,E1)
    tr.put_weights(W1)
    res1 = tr.WienerIndex()
    print(res1)


if __name__ == "__main__":
    test()
