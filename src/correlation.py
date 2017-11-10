import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cossim
from scipy.stats import pearsonr

def getsimmatrix(embvectors,N,embchars):
    corrmat = np.zeros((len(embchars),len(embchars)))

    for i, e in enumerate(embvectors):        
        if not i in embchars:
            continue
        for j, d in enumerate(embvectors):
            if not j in embchars:
                continue
            corrmat[i][j] = cossim(e.reshape(1,-1),d.reshape(1,-1))
    return corrmat

def correlation(x,y):
    assert(x.shape == y.shape)

    xl = []
    yl = []

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xl.append(x[i][j])
            yl.append(y[i][j])
    return pearsonr(xl,yl)
