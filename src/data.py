from collections import Counter

from features import transform

# Indices for wordform, lemma and tags in the data list.
WF    = 1
LEMMA = 2
TAGS  = 3

# Minimum number of character occurrences we require for embedded
# characters.
MINCHAROCC = 100

# Word boundary.
WB = '#'

def encode(x,encoder):
    encoder.setdefault(x,len(encoder))
    return encoder[x]

def count(s,counter):
    for x in s:
        counter[x] += 1

def readdata(fn,lan):
    data = []
    charcounts = Counter()

    # We need to first count all characters to make sure that
    # characters occurring more than MINCHAROCC will get the first 0
    # ... N character codes.
    for line in map(lambda l: l.strip('\n'), open(fn)):
        wf, tags, lemma = line.lower().split('\t')
        wf = transform(wf,lan)
        lemma = transform(lemma,lan)

        count(wf,charcounts)
        data.append((wf,lemma,tags))

    charcounts = sorted([(count,char) for char, count in charcounts.items()],reverse=1)
    tagencoder = {}
    cencoder = {x[1]:i for i, x in enumerate(charcounts)}

    for i, d in enumerate(data):
        lemma, wf, tags = d
        wf = [encode(c,cencoder) for c in WB + wf + WB]
        tags = [encode(t,tagencoder) for t in tags.split(',')]
        lemma = [encode(c,cencoder) for c in WB + lemma + WB]
        data[i] = (lemma,wf,tags)
    embedchars = set([cencoder[c] for count,c in charcounts if count > MINCHAROCC])
    return data, cencoder, tagencoder, embedchars
            
