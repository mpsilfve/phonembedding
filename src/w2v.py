import numpy as np

from gensim.models import Word2Vec

WINDOW=1
SG=1
HS=0
MINCOUNT=100
NS=3

def getw2v(data,embchars,dim,cdecoder):
    # Encode ids to strings for gensim implementation of w2v.
    words = [[str(c) for c in wf] for wf, lemma, tags in data]
    lemmas = [[str(c) for c in lemma] for wf, lemma, tags in data]
    model = Word2Vec(sentences=words+lemmas,
                     size=dim,
                     window=WINDOW,
                     sg=SG,
                     hs=HS,
                     min_count=MINCOUNT,
                     negative=NS)
    m = np.zeros((len(embchars),dim))
    for i in range(len(embchars)):
        m[i] = model[str(i)]
    return m
