
from loaders import Token
from collections import defaultdict

import numpy as np

import re
import unicodedata

UNK = "$$UNK$$" # todo: unknown form with known morpho

isPunct = lambda w: all([l != "/" and unicodedata.category(l)=="Po" for l in w])
isNum = lambda w: re.match("^[0-9_ .,]+$",w) is not None

class FeatureSampler:

    def __init__(self, counts):
        feats = []
        positions = [0]
        i = 0
        for f,c in counts.items():
            feats.append(f)
            positions.append(c+i)
            i += c
        self.feats = feats
        self.counts = np.array(positions)
        self.total = i

    def sample_one(self, positives= None, limit = 3):
        if limit == 0:
            return -1
        i = np.random.randint(0, self.total)
        f = self.feats[self.counts.searchsorted(i, side='right') - 1]
        if positives is not None:
            if f in positives:
                f = self.sample_one(positives, limit -1)
        return f

class Encoder:

    def __init__(self, data, function_tags, imax, words_counts, counts, minOcc, revIndex):
        self.data = data
        self.function_tags = function_tags
        self.imax = imax
        self.nforms = len(data["form"])
        self.words_counts = words_counts
        self.subsampling = self.compute_subsampling(words_counts)
        self.feature_samplers = self.revindex_from_counts(counts)
        self.minOcc = minOcc
        self.sent_count = 0
        self.revIdx = revIndex


    def compute_subsampling(self, wc, t=0.0001):
        ntok = sum(wc.values())
        subsampling_probability = {}
        for k, v in wc.items():
            f = v / ntok
            p = (t / f) * (1 + np.sqrt(f / t))
            subsampling_probability[k] = p
        return subsampling_probability

    def revindex_from_counts(self, counts):
        idx = {}
        for kind, d in counts.items():
            idx[kind] = FeatureSampler(d)
        return idx

    def encode_morpheme(self, morpheme, relative_position=0):
        pos = ""
        if relative_position == 1:
            pos = "next_"
        if relative_position == -1:
            pos = "prev_"
        kind = pos + "morpho"
        return self.data[kind][morpheme]

    def encode_form(self, form, relative_position=0):
        if relative_position == 1:
            return self.data["next_form"][form]
        elif relative_position == -1:
            return self.data["prev_form"][form]
        else:
            return self.data["form"][form]

    def encode_tag(self, tag, relative_position=0):
        if relative_position == 1:
            return self.data["next_tag"][tag]
        elif relative_position == -1:
            return self.data["prev_tag"][tag]
        else:
            return self.data["tag"][tag]

    def find_function_word_code(self, sentence, position, step, max_steps=1):
        if max_steps == 0:
            return None
        if step == -1:
            kind = "prev_function_word"
        elif step == 1:
            kind = "next_function_word"
        else:
            raise ValueError("step should be 1 or -1")
        position += step
        if position < 0 or position >= len(sentence):
            return None
        tok = sentence[position]
        if tok.tag in self.function_tags:
            return self.data[kind][tok.form]
        else:
            return self.find_function_word_code(sentence, position, step, max_steps - 1)

    def get_features_at(self, sentence, position):
        tok = sentence[position]
        p = np.random.uniform()
        if p > self.subsampling[tok.form]:
            # print("discard", p, tok.form)
            return []
        # yield self.encode_form(tok.form)
        #yield self.encode_tag(tok.tag)
        for morph in tok.morphs:
            yield self.encode_morpheme(morph)

        if position > 0:
            prevtok = sentence[position - 1]
            yield self.encode_form(prevtok.form, -1)
            yield self.encode_tag(prevtok.tag, -1)
            for morph in prevtok.morphs:
                yield self.encode_morpheme(morph, -1)

        if position + 1 < len(sentence):
            next_tok = sentence[position + 1]
            yield self.encode_form(next_tok.form, +1)
            yield self.encode_tag(next_tok.tag, +1)
            for morph in next_tok.morphs:
                yield self.encode_morpheme(morph, +1)
        if tok.tag not in self.function_tags:
            pf = self.find_function_word_code(sentence, position, -1)
            if pf is not None:
                yield pf
            nf = self.find_function_word_code(sentence, position, +1)
            if nf is not None:
                yield nf

        # boolean features
        yield self.data["isPunct"][isPunct(tok.form)]
        yield self.data["isNum"][isNum(tok.form)]

    def sample_negative_features(self, positives):
        # for k in ["form", "next_form", "prev_form", "next_tag", "prev_tag", "morpho", "prev_morpho", "next_morpho", "prev_function_word", "next_function_word"]:
        # for k in ["tag", "next_form", "prev_form", "next_tag", "prev_tag", "morpho", "prev_morpho", "next_morpho", "prev_function_word", "next_function_word"]:
        for k in ["next_form", "prev_form", "next_tag", "prev_tag", "morpho", "prev_morpho", "next_morpho", "prev_function_word", "next_function_word"]:
        # for k in ["form", "next_form", "prev_form", "morpho-pfx", "morpho-sfx"]:
            i = self.feature_samplers[k].sample_one(positives)
            if i != -1:
                yield i #self.data[k][i]


    def get_random_features(self, n=4):
        for f in np.random.randint(0, self.imax, n):
            yield f

    def positive_contexts_builder(self, sentence):
        for i, tok in enumerate(sentence):
            if tok.form != UNK:
                current = self.encode_form(tok.form)
                for f in self.get_features_at(sentence, i):
                    yield (current, f)


    def negative_contexts_builder(self, sentence, n, positives):
        for i, tok in enumerate(sentence):
            if tok.form != UNK:
                current = self.encode_form(tok.form)
                for _ in range(n):
                    for f in self.sample_negative_features(positives):
                        yield (current, f)
                # boolean features
                    yield (current, self.data["isPunct"][not isPunct(tok.form)])
                    yield (current, self.data["isNum"][not isNum(tok.form)])


    def contexts_builder(self, sentence, nb_neg ):
        # self.sent_count += 1
        # print(self.sent_count)
        sentence = [Token(UNK, tok.morphs, tok.tag) if self.words_counts[tok.form] < self.minOcc else tok for tok in sentence]
        # sentence = [tok for tok in sentence if self.words_counts[tok.form] >= self.minOcc]
        data = list(self.positive_contexts_builder(sentence))
        positives = set([f for _, f in data])
        labels = [1] * len(data)
        negs = list(self.negative_contexts_builder(sentence, nb_neg, positives))
        data.extend(negs)
        labels.extend([0] * len(negs))
        return data, labels

    def decodeContextsList(self, data, labels = None):
        return [ (self.revIdx["forms"][w], self.revIdx["feats"][feat],l)  for ((w, feat),l) in zip(data, labels) if l ==1]

def build_encoder(corpus, function_tags, minOcc=10):
    text_data = defaultdict(lambda : defaultdict(int))
    wc = defaultdict(float)
    ntok = 0.0


    # basic features
    for line in corpus:
        for i, token in enumerate(line):
            text_data["form"][token.form] += 1
            wc[token.form] += 1.0
            ntok += 1.0
            for morph in token.morphs:
                text_data["morpho"][morph] += 1
            text_data["tag"][token.tag] += 1
            if token.tag in function_tags:
                text_data["prev_function_word"][token.form] += 1

    # insert UNK
    text_data["form"][UNK] = minOcc # sum([c for c in text_data["form"].values() if c < minOcc])
    wc[UNK] = text_data["form"][UNK]
    text_data["prev_function_word"][UNK] = minOcc # sum([c for c in text_data["prev_function_word"].values() if c < minOcc])
    text_data["form"] = {k:v for k,v in text_data["form"].items() if v >= minOcc}

    # contextual features
    text_data["prev_form"] = text_data["form"].copy()
    text_data["prev_tag"] = text_data["tag"].copy()
    text_data["prev_morpho"] = text_data["morpho"].copy()

    text_data["next_form"] = text_data["form"].copy()
    text_data["next_tag"] = text_data["tag"].copy()
    text_data["next_morpho"] = text_data["morpho"].copy()

    text_data["next_function_word"] = text_data["prev_function_word"].copy()

    # build mapping to integers
    index = {}
    revIndex = {"forms":{}, "feats":{}}
    i = 0
    index["form"] = {}
    for f in text_data["form"]:
        index["form"][f] = i
        revIndex["forms"][i] = "form:" + f
        i += 1
    i = 0
    for k, s in text_data.items():
        if k != "form":
            index[k] = {}
            for v in s:
                if v == "":
                    print(k)
                index[k][v] = i
                revIndex["feats"][i] = k + ":"+v
                i += 1
    # ajout des tests booleens
    index["isNum"] = {True: i, False: i+1}
    index["isPunct"] = {True: i+2, False: i+3}
    revIndex["feats"][i] = "isNum:True"
    revIndex["feats"][i+1] = "isNum:False"
    revIndex["feats"][i+2] = "isPunct:True"
    revIndex["feats"][i+3] = "isPunct:False"
    i += 4

    #version pré-indexée des comptes
    idxCounts = {}
    for k, v in text_data.items():
        idxCounts[k] = {}
        for txt, count in v.items():
            idxCounts[k][index[k][txt]] = count


    # computing sub-sampling (keeping) probability
    return Encoder(index, function_tags, i, wc, idxCounts, minOcc, revIndex)
