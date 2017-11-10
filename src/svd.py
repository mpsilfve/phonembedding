from sklearn.metrics.pairwise import cosine_similarity as cossim
import numpy as np

WINDOW = 2

def buildmatrix(data,charencoder):
    # Initializes to non-zero to avoid NAN during np.log.
    jointcounts = np.ones((len(charencoder), len(charencoder))) * 0.0001
    singlecounts = np.ones((len(charencoder), 1)) * 0.0001

    
    jointtot = 0
    singletot = 0

    for wf, _, _ in data:
        for i, c in enumerate(wf):
            for j in range(i - WINDOW, i + WINDOW):
                if j == i or j < 0 or j >= len(wf):
                    continue
                jointcounts[c][wf[j]] += 1
                jointtot += 1
            singlecounts[c][0] += 1
            singletot += 1

    jointdistr = jointcounts * (1.0 / jointtot)
    singledistr = singlecounts * (1.0 / singletot)
    pmi = np.log(np.divide(jointdistr,np.dot(singledistr,
                                             np.transpose(singledistr))))
    return np.multiply(pmi,pmi > 0)



def getsvd(data, charencoder):
    ppmimatrix = buildmatrix(data,charencoder)
    u, s, vt = np.linalg.svd(ppmimatrix)
    return np.dot(u, np.diag(s))

def truncate(m,d):
    return m[0:m.shape[0],0:d]
