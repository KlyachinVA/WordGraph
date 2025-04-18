from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import pickle

def ByNB(fname):
    data = np.load(fname)
    X = data["X"]
    Y = data["Y"]
    XX = []
    YY = []
    D = []
    N = len(X)
    # k=2
    k = 6
    print(f"N={N}")
    for i in range(N):
        if X[i,0] != np.inf:
            # print(Y[i])
            # D.append([X[i, 0], X[i, 1],  Y[i]])
            # D.append([X[i,0],X[i,1],X[i,2],Y[i]])
            D.append([X[i, 0], X[i, 1], X[i, 2], X[i,3], X[i,4], X[i,5], Y[i]])
    random.shuffle(D)
    N = len(D)
    n = int(2*N/3)
    print(f"New N={N}")
    print(f"n = {n}")
    D = np.array(D)
    XX = D[:,:k]
    YY = D[:,k]
    X_train = XX[:n]
    Y_train = YY[:n]

    X_test = XX[n:]
    Y_test = YY[n:]

    Nt = len(Y_test)
    print(f"For test data Nt={Nt}")




    # model = GaussianNB() # 72%
    # model = BernoulliNB() # 50%
    # model = MultinomialNB() # 45%
    # model = SVC() # 67%
    # model = KNeighborsClassifier(n_neighbors=5) # 72%
    # model = DecisionTreeClassifier() # 72%
    # model = LinearDiscriminantAnalysis() # 73%
    # model = RandomForestClassifier() # 72%

    need_fit = True
    fname_model = "./data/models/simple.bin"
    if need_fit:
        f = open(fname_model,"wb")
        model.fit(X_train,Y_train)
        pickle.dump(model,f)
        f.close()
    else:
        f = open(fname_model, "rb")
        model = pickle.load(f)
        f.close()

    Y_pred = model.predict(X_test)
    print(Y_pred)
    print(Y_test)

    Np = (Y_test - Y_pred == 0).sum()

    print(f"Точность = {100*Np/Nt}%")

def test():
    fname = "./results/inform-dataset-6.npz"
    ByNB(fname)


if __name__ == "__main__":
    test()