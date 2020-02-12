#import pyximport
#pyximport.install(pyimport = True)

import numpy as np
import sys
import os.path
#np.random.seed(13)

import glob
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Flatten, merge
from keras.layers.merge import Dot
from keras import regularizers
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from joblib import Parallel, delayed

# import gensim

from loaders import load_conll, load_morpho, lexicons_of_corpus, generate_batch
# from tsne import reduce_dimensions
from features import Encoder, build_encoder



corpus_file = sys.argv[1]
morpho_file = sys.argv[2]
minOcc = int(sys.argv[3])
dim_list = sys.argv[4].split(",")
output_dir = sys.argv[5]


neg_factors = [5] #, 5, 20, 30]
deltaMin = 10.0  # will divide first loss to get actual deltaMin

DUMP_EVERY = 200

def save_vectors(SkipGram, encoder, dataDir, dim_embeddings, nb_negs, iteration):
    print("saving vectors")
    vectors = SkipGram.get_weights()[0]
    f = open(dataDir + '/{}d-{}n-{}i.vec'.format(dim_embeddings, 5, iteration), 'w')
    f.write('{} {}\n'.format(encoder.nforms - 1, dim_embeddings))
    # todo: en faire une fonction dans l'encodeur
    for word, i in encoder.data["form"].items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()


if __name__ == "__main__": # for corpus_file in glob.glob(dataDir): # todo: option
    # prop_train = corpus_file.rsplit("-",1)[1][:-4]
    print("loading Morpho",flush=True)
    morpho = load_morpho(morpho_file)
    print("loaded, loading corpus..", flush=True)
    corpus = load_conll(corpus_file, morpho)
    print("loaded, building encoder", flush=True)
    # encoder = build_encoder(corpus, ["ADP", "DET", "CONJ", "PRON"])
    encoder = build_encoder(corpus, ["DT", "CONJ", "C", "NEG", "PREP","PRO","T", "ADP", "DET", "CONJ", "PRON", "CCONJ", "SCONJ", "ADPDET"], minOcc= minOcc)
    print("built", flush=True)
    for nb_negs in neg_factors:
        batchs = []
        print("creating data")
        print("{} sentences".format(len(corpus)), flush=True)
        #allSamples = Parallel(n_jobs=-1, batch_size=10000)(delayed(Encoder.contexts_builder)(encoder, s, nb_negs) for s in corpus)
        allSamples = [encoder.contexts_builder(s,nb_negs) for s in corpus]
        print("shuffling samples", flush=True)
        np.random.shuffle(allSamples)
        for d,l in allSamples[:3]:
            print(encoder.decodeContextsList(d,l))
            print("\n")
        print("creating batchs")
        data_size = 0
        for data, labels in generate_batch(16, allSamples): # 512 ?
            x = [np.array(x, dtype=np.float32) for x in zip(*data)]
            y = np.array(labels, dtype=np.float32)
            # print(len(y))
            data_size += len(y)
            batchs.append((x, y))
        print(data_size)
        print(x)
        print(y)
        for dim_embeddings in [int(x) for x in dim_list]:
            
            # inputs
            w_inputs = Input(shape=(1, ), dtype=np.float32)
            w = Embedding(encoder.nforms, dim_embeddings, # embeddings_regularizer=regularizers.l2(0.01),
                          embeddings_initializer='glorot_normal')(w_inputs)
            #w = Embedding(encoder.nforms, dim_embedddings)(w_inputs)
            
            # context
            c_inputs = Input(shape=(1, ), dtype=np.float32)
            c = Embedding(encoder.imax, dim_embeddings, # embeddings_regularizer=regularizers.l2(0.01),
                          embeddings_initializer='glorot_normal')(c_inputs)
            #c = Embedding(encoder.imax, dim_embedddings)(c_inputs)


            o = Dot(axes=2)([w, c])
            o = Reshape((1,), input_shape=(1, 1))(o)
            o = Activation('sigmoid')(o)
            #o = Activation('tanh')(o)
            #o = Activation('relu')(o)
            # todo: tester la régularisation

            SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
            SkipGram.summary()
            #SkipGram.compile(loss='squared_hinge', optimizer='adam')
            SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

            print("training network")
            delta = 1000000
            previous_loss = None
            it = 0
            while it < 50: # or delta > deltaMin:  # todo: option
                loss = 0.
                losses = []
                np.random.shuffle(batchs)
                for x, y in batchs:
                    if x:
                        #tmp = zip(x[0],x[1])
                        #print(data)
                        #print(y)
                        #print(encoder.decodeContextsList(list(tmp)[:50], list(y[:50])))
                        losses.append(SkipGram.train_on_batch(x, y))
                loss = np.mean(losses)
                if previous_loss:
                    delta = previous_loss - loss
                if it == 0:
                    deltaMin = min(0.001,loss / deltaMin) # 0.001 ?
                    print("deltaMin: ", deltaMin)
                it += 1
                if it % DUMP_EVERY == 0:
                    save_vectors(SkipGram, encoder, output_dir, dim_embeddings, nb_negs, it)
                previous_loss = loss
                print(loss, flush=True)
            save_vectors(SkipGram, encoder, output_dir, dim_embeddings, nb_negs, "last")

# reduction de dim pour visu
#print("TSNE")
#dims = 2
#small = reduce_dimensions(vectors, dims, metric='cosine')
#f = open('small.txt', 'w')
#f.write('{} {}\n'.format(lex.nforms - 1, dims))
#for word, i in lex.forms.items():
#    f.write('{} {}\n'.format(word, ' '.join(map(str, list(small[i, :])))))
#f.close()
#
# w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
