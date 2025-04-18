from scipy.sparse.csgraph import connected_components
import numpy as np

def test():
    W = np.array([[0,1,0,0,0,0],
                  [1,0,1,0,0,0],
                  [0,1,0,0,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,1,0,1],
                  [0,0,0,0,1,0],])
    k,lbs = connected_components(W,False,return_labels=True)
    print(k)
    print(lbs)


if __name__ == "__main__":
    test()