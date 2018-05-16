from collections import defaultdict
from collections import namedtuple
from codecs import open
import os

import numpy as np
from sklearn.externals import joblib


TAG = "\ue00a"
MORPH = "\ue00b"


def load_morpho(path):
    data = defaultdict(lambda x: set(x))
    with open(path, "r", "utf8") as file:
        for line in file:
            (w, morphs) = line.split("\t", 1)
            s = set(morphs.strip().split(MORPH))
            if len(s) < 2:
                s = set(["%%"+w, w+"%%"])
            data[w] = s
    return data

Token = namedtuple("Token", "form morphs tag")


def load_conll(path, morpho):
    data = [[]]
    with open(path, "r", "utf8") as file:
        for line in file:
            fields = line.strip().split("\t")
            if len(fields) == 4:
                form = fields[1]
                morphs = morpho.get(form, set(["%%" + form, form + "%%"]))
                tag = fields[3]
                data[-1].append(Token(form, morphs, tag))
            elif len(fields) < 4:
                data.append([])
            else:
                form = fields[1]
                tag = fields[3]
                morphs = morpho.get(form,set(["%%form", "form%%"]))
                data[-1].append(Token(form, morphs, tag))
    return data

def load_corpus(path):
    data = []
    with open(path, "r", "utf8") as file:
        for line in file:
            if(chr(0x85) in line):
                continue

            if line.strip() == "":
                continue
            else:
                tokens = [tok.rsplit(TAG, 1) for tok in line.strip().split(" ")]
                assert(len(tokens) > 0)
                if(any([len(tok) != 2 for tok in tokens])):
                    print(tokens)
                assert(all([len(tok) == 2 for tok in tokens]))
                tokens = [(w, t) for w, t in tokens if len(w) > 0]
                assert (all([len(m) > 0 and len(t) > 0 for w, t in tokens for m in w.split(MORPH)]))
                if(any([len(w.split(MORPH))!= 2 for w, t in tokens])):
                    # print("pb", line)
                    # print([w for w, t in tokens if len(w.split(MORPH)) != 2])
                    tokens = [(w, t) if MORPH in w else (w+MORPH+"-0", t) for w, t in tokens]

                data.append([Token(w.replace(MORPH, "").replace("-0", ""),
                                   w.split(MORPH), t)
                             for w, t in tokens])
    print(len(data))
    return data


def build_tag_lexicon_conll(path, V=None):
    lex = defaultdict(set)
    tagset = set()
    counts = defaultdict(int)
    for sentence in load_conll(path, defaultdict(set)):
        for tok in sentence:
            lex[tok.form].add(tok.tag)
            tagset.add(tok.tag)
            counts[tok.form] += 1
    tagset = sorted(list(tagset))
    tagidx = {t: i for i, t in enumerate(tagset)}
    if V is not None:
        keys = set(sorted(counts, key=lambda x:counts[x])[-V:])
        lex = {k:lex[k] for k in keys if k in lex}
    for k, v in lex.items():
        lex[k] = set(v)
    return lex, tagset

def build_tag_lexicon(path, V=None):
    lex = defaultdict(set)
    tagset = set()
    counts = defaultdict(int)
    for sentence in load_corpus(path):
        for tok in sentence:
            lex[tok.form].add(tok.tag)
            tagset.add(tok.tag)
            counts[tok.form] += 1
    tagset =  sorted(list(tagset))
    tagidx = {t:i for i,t in enumerate(tagset)}
    if V is not None:
        keys = set(sorted(counts, key=lambda x:counts[x])[-V:])
        lex = {k:lex[k] for k in keys if k in lex}
    for k, v in lex.items():
        lex[k] = set([tagidx[t] for t in v])
    return lex, tagset


Lex = namedtuple("Lex", "nforms forms roots pfx sfx tags ntags size")
def lexicons_of_corpus(corpus):
    forms = {}
    nforms = 0
    roots = {}
    nroots = 0
    pfx = {}
    npfx = 0
    sfx = {}
    nsfx = 0
    tags = {}
    ntags = 0

    for line in corpus:
        for tok in line:
            f = tok.form
            if f not in forms:
                forms[f] = nforms
                nforms += 1
            if len(tok.morphs) == 2:
                l, r = tok.morphs
                if r != "-0":
                    if len(l) <= len(r):
                        if l not in pfx:
                            pfx[l] = npfx
                            npfx += 1
                        root = r
                    else:
                        if r not in sfx:
                            sfx[r] = nsfx
                            nsfx += 1
                        root = l
                else:
                    root = tok.form
            else:
                root = tok.form
            if root not in roots:
                roots[root] = nroots
                nroots += 1
            if tok.tag not in tags:
                tags[tok.tag] = ntags
                ntags += 1

    for (k, v) in roots.items():
        roots[k] += nforms * 2
    for (k, v) in pfx.items():
        pfx[k] += nforms * 2 + nroots
    for (k, v) in sfx.items():
        sfx[k] += nforms * 2 + nroots + npfx
    for (k, v) in tags.items():
        tags[k] += nforms * 2 + nroots + npfx + nsfx

    return Lex(nforms, forms, roots, pfx, sfx, tags, ntags, nforms*2 + nroots + npfx + nsfx + ntags*2)



def generate_batch(size, seq):
    buff = [[], []]
    n = 0
    for data, labels in seq:
        buff[0].extend(data)
        buff[1].extend(labels)
        n += 1
        if n >= size:
            yield buff
            buff = [[], []]
            n = 0
    if n >0:
        yield buff


def load_space(path, lex):
    with open(path) as file:
        l1 = file.readline()
        fields = l1.strip().split(" ")
        assert(len(fields) == 2)
        N = int(fields[0]) + 1
        dims = int(fields[1])
        data = []
        i = 0
        index = {}
        rindex = []
        for line in file.readlines():
            fields = line.strip().split(" ")
            w = fields[0]
            if lex is None or w in lex:
                vect = [float(x) for x in fields[1:]]
                data.append(vect)
                index[w] = i
                rindex.append(w)
                i += 1
    r = np.empty((i, dims))
    for i,v in enumerate(data):
        r[i] = v
    return r, index, rindex


def load_from_yaset(path):
    matrix = joblib.load(os.path.join(path, "embeddings.pkl"))
    word_mapping = joblib.load(os.path.join(path, "word_mapping.pkl"))
    ridx = [-1] * len(word_mapping)
    for (w, i) in word_mapping.items():
        ridx[i] = w
    return matrix[0], word_mapping, ridx
