from sys import stderr, argv

import dynet as dy
import random
import pickle

LSTM_NUM_OF_LAYERS = 1
STATE_SIZE = 32

EPOCHS=10

def init(embedding_size,cencoder,tencoder):
    global model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, char_lookup, attention_w1,\
    attention_w2,attention_v,decoder_w,decoder_b,output_lookup, tag_lookup
    model = dy.Model()
    
    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, embedding_size, STATE_SIZE, 
                                  model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, embedding_size, STATE_SIZE, 
                                  model)
    
    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+embedding_size, 
                              STATE_SIZE, model)
    char_lookup = model.add_lookup_parameters((len(cencoder), embedding_size))
    tag_lookup = model.add_lookup_parameters((len(tencoder), embedding_size))
    decoder_w = model.add_parameters( (len(cencoder), STATE_SIZE))
    decoder_b = model.add_parameters( (len(cencoder)))

def embed_sentence(lemma,tags):
    return [char_lookup[char] for char in lemma] + [tag_lookup[tag] for tag in tags]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def decode(dec_lstm, vectors, output,cencoder):
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    encoded = vectors[-1]
    w1dt = None

    last_output_embeddings = char_lookup[cencoder["#"]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        vector = dy.concatenate([encoded,last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = char_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(lemma, tag, enc_fwd_lstm, enc_bwd_lstm, dec_lstm,cencoder,cdecoder):
    embedded = embed_sentence(lemma,tag)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    encoded = encoded[-1]
    w1dt = None

    last_output_embeddings = char_lookup[cencoder["#"]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(lemma)*2):
        if count_EOS == 2: break
        vector = dy.concatenate([encoded,last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = char_lookup[next_char]
        if cdecoder[next_char] == "#":
            count_EOS += 1
            continue

        out += cdecoder[next_char]
    return out


def get_loss(lemma, tags, wf, enc_fwd_lstm, enc_bwd_lstm, dec_lstm,cencoder):
    dy.renew_cg()
    embedded = embed_sentence(lemma,tags)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, wf,cencoder)


def train(data,cencoder,cdecoder,tencoder,tdecoder,embedding_size):
    init(embedding_size,cencoder,tencoder)
    trainer = dy.SimpleSGDTrainer(model)
    for n in range(EPOCHS):
        totalloss = 0
        random.shuffle(data)
        for i, io in enumerate(data):
            print('EPOCH %u: ex %u of %u\r' % (n+1,i+1,len(data)),end='',file=stderr)
            lemma,wf,tags = io
            loss = get_loss(lemma, tags, wf, enc_fwd_lstm, enc_bwd_lstm, dec_lstm,cencoder)
            totalloss += loss.value()
            loss.backward()
            trainer.update()
        print()
        print(totalloss/len(data),file=stderr)
        for lemma,wf,tags in data[:10]:
            print('input:',''.join([cdecoder[c] for c in lemma]+[tdecoder[c] for c in tags]),
                  'sys:',generate(lemma,tags, enc_fwd_lstm, enc_bwd_lstm, dec_lstm,
                                  cencoder,cdecoder),
                  'gold:',''.join([cdecoder[c] for c in wf]),file=stderr)
    return char_lookup.rows_as_array(list(cdecoder.keys()))                     

if __name__=='__main__':
    data = readdata(argv[1])
    train(model, data)
    model.save(argv[2])    
    pickle.dump((int2char,char2int,VOCAB_SIZE),open("%s.obj.pkl" % argv[2],"wb"))
