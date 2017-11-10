from copy import copy

from torch import FloatTensor, LongTensor, randn, stack, cat
from torch import max as tmax
from torch.autograd import Variable
from torch.nn import Embedding, Sequential, Linear, LogSoftmax, NLLLoss, LSTM, ModuleList
from torch.optim import Adam

import numpy as np

DTYPE=FloatTensor
LTYPE=LongTensor
LSTMDIM=32
MAXWFLEN=40
LOSS=NLLLoss()
LEARNINGRATE=0.001
BETAS=(0.9,0.9)
EPOCHS=20

BD="<#>"

def initmodel(cencoder,tencoder,embdim):
    cencoder = copy(cencoder)
    tencoder = copy(tencoder)
    tencoder[BD] = len(tencoder)
    cencoder[BD] = len(cencoder)
    cembedding=Embedding(len(cencoder),embdim)
    tembedding=Embedding(len(tencoder),embdim)
    enc = LSTM(input_size=embdim,
               hidden_size=LSTMDIM,
               num_layers=1,
               bidirectional=1).type(DTYPE)
    ench0 = randn(2, 1, LSTMDIM).type(DTYPE)
    encc0 = randn(2, 1, LSTMDIM).type(DTYPE)

    dec = LSTM(input_size=2*LSTMDIM+embdim,
               hidden_size=LSTMDIM,
               num_layers=1).type(DTYPE)
    dech0 = randn(2, 1, 2*LSTMDIM+embdim).type(DTYPE)
    decc0 = randn(2, 1, 2*LSTMDIM+embdim).type(DTYPE)

    pred = Linear(LSTMDIM,len(cencoder)).type(DTYPE)
    sm = LogSoftmax().type(DTYPE)

    model = ModuleList([cembedding,
                        tembedding,
                        enc,
                        dec,
                        pred,
                        sm])
    optimizer = Adam(model.parameters(),
                     lr=LEARNINGRATE,
                     betas=BETAS)

    return {'model':model,
            'optimizer':optimizer,
            'cencoder':cencoder,
            'tencoder':tencoder,
            'cembedding':cembedding,
            'tembedding':tembedding,
            'enc':enc,
            'ench0':ench0,
            'encc0':encc0,
            'dec':dec,
            'dech0':dech0,
            'decc0':decc0,
            'pred':pred,
            'sm':sm,
            'embdim':embdim}

def encode(lemma,tags,modeldict):
    lemma = [modeldict['cencoder'][BD]] + lemma
    tags = tags + [modeldict['cencoder'][BD]]
    lemmalen=len(lemma)
    tagslen=len(tags)
    lemma = Variable(LTYPE(lemma),requires_grad=0)
    tags = Variable(LTYPE(tags),requires_grad=0)
    lemma = modeldict['cembedding'](lemma)
    tags = modeldict['tembedding'](tags)
    
    all = cat([lemma,tags],dim=0)
    _, finals = modeldict['enc'](all.view(lemmalen+tagslen,1,modeldict['embdim']),
                                 (Variable(modeldict['ench0'],requires_grad=1),
                                  Variable(modeldict['encc0'],requires_grad=1)))
    finalh0, finalc0 = finals
    return finalc0

def decode(encoded,modeldict,wflen=MAXWFLEN):
    pchar = modeldict['cencoder'][BD]
    h = Variable(modeldict['ench0'],requires_grad=1)
    c = Variable(modeldict['encc0'],requires_grad=1)
    cdistrs = []
    chars = []
    for i in range(min(wflen+1,MAXWFLEN)):
        pchar = Variable(LTYPE([pchar]),requires_grad=0)
        pemb = modeldict['cembedding'](pchar)
        input = cat([pemb,encoded.view(1,2*LSTMDIM)],dim=1)
        _, states = modeldict['dec'](input,(h,c))
        h,c = states
        c = c.view(1,LSTMDIM)
        cdistr = modeldict['sm'](modeldict['pred'](c))
        cdistrs.append(cdistr)
        _, pchar = tmax(cdistr,1)
        pchar = int(pchar.data.numpy()[0])
        chars.append(pchar)
    return cat(cdistrs,dim=0),chars

def update(lemma,tags,wf,modeldict):
    encoded = encode(lemma,tags,modeldict)
    cdistrs, chars = decode(encoded,modeldict,len(wf))
    cdistrs = cdistrs[:len(chars)]
    chars = Variable(LTYPE(chars),requires_grad=0)
    loss = LOSS(cdistrs,chars)
    lossval = loss.data[0]
    loss.backward()
    modeldict['optimizer'].step()
    return lossval

def train(data,modeldict):
    for i in range(EPOCHS):
        totloss = 0
        for i, wlt in enumerate(data):
            wf, lemma, tags = wlt
            totloss += update(lemma,tags,wf,modeldict)
            print(i+1," ",len(data)," ",totloss/(i+1))
