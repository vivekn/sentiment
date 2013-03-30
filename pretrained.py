"""
Train a naive Bayes classifier from the IMDb reviews data set
"""
from __future__ import division
from collections import defaultdict
from math import log, exp
from functools import partial
import re
import os
import random
import pickle
import pylab


handle = open("trained", "rb")
sums, positive, negative = pickle.load(handle)

def tokenize(text):
    return re.findall("\w+", text)

def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    for word in words:
        stripped = word.strip(delims).lower()
        result.append("not_" + stripped if negation else stripped)

        if any(neg in word for neg in frozenset(["not", "n't", "no"])):
            negation = not negation

        if any(c in word for c in delims):
            negation = False
    return result

def get_positive_prob(word):
    return 1.0 * (positive[word] + 1) / (2 * sums['pos'])

def get_negative_prob(word):
    return 1.0 * (negative[word] + 1) / (2 * sums['neg'])

def classify(text, pneg = 0.5, preprocessor=negate_sequence):
    words = preprocessor(text)
    pscore, nscore = 0, 0

    for word in words:
        pscore += log(get_positive_prob(word))
        nscore += log(get_negative_prob(word))

    return pscore > nscore

def classify_demo(text):
    words = negate_sequence(text)
    pscore, nscore = 0, 0

    for word in words:
        pdelta = log(get_positive_prob(word))
        ndelta = log(get_negative_prob(word))
        pscore += pdelta
        nscore += ndelta
        print "%25s, pos=(%10lf, %10d) \t\t neg=(%10lf, %10d)" % (word, pdelta, positive[word], ndelta, negative[word]) 

    print "\nPositive" if pscore > nscore else "Negative"
    print "Confidence: %lf" % exp(abs(pscore - nscore))
    return pscore > nscore, pscore, nscore

def test():
    strings = [
    open("pos_example").read(), 
    open("neg_example").read(),
    "This book was quite good.",
    "I think this product is horrible."
    ]
    print map(classify, strings)

def mutual_info(word):
    """
    Finds the mutual information of a word with the training set.
    """
    cnt_p, cnt_n = sums['pos'], sums['neg']
    total = cnt_n + cnt_p
    cnt_x = positive[word] + negative[word]
    if (cnt_x == 0): 
        return 0
    cnt_x_p, cnt_x_n = positive[word], negative[word]
    I = [[0]*2]*2
    I[0][0] = (cnt_n - cnt_x_n) * log ((cnt_n - cnt_x_n) * total / cnt_x / cnt_n) / total 
    I[0][1] = cnt_x_n * log ((cnt_x_n) * total / (cnt_x * cnt_n)) / total if cnt_x_n > 0 else 0
    I[1][0] = (cnt_p - cnt_x_p) * log ((cnt_p - cnt_x_p) * total / cnt_x / cnt_p) / total 
    I[1][1] = cnt_x_p * log ((cnt_x_p) * total / (cnt_x * cnt_p)) / total if cnt_x_p > 0 else 0

    return sum(map(sum, I))

def reduce_features(features, stream):
    return [word for word in negate_sequence(stream) if word in features]

def feature_selection_experiment(test_set):
    """
    Select top k features. Vary k from 1000 to 50000 and plot data
    """
    keys = positive.keys() + negative.keys()
    sorted_keys = sorted(keys, cmp=lambda x, y: mutual_info(x) > mutual_info(y)) # Sort descending by mutual info
    features = set()
    num_features, accuracy = [], []
    print sorted_keys[-100:]

    for k in xrange(0, 50000, 1000):
        features |= set(sorted_keys[k:k+1000])
        preprocessor = partial(reduce_features, features)
        correct = 0
        for text, label in test_set:
            correct += classify(text) == label
        num_features.append(k+1000)
        accuracy.append(correct / len(test_set))
    print negate_sequence("Is this a good idea")
    print reduce_features(features, "Is this a good idea")

    pylab.plot(num_features, accuracy)
    pylab.show()

def get_paths():
    """
    Returns supervised paths annotated with their actual labels.
    """
    posfiles = [("./aclImdb/test/pos/" + f, True) for f in os.listdir("./aclImdb/test/pos/")[:500]]
    negfiles = [("./aclImdb/test/neg/" + f, False) for f in os.listdir("./aclImdb/test/neg/")[:500]]
    return posfiles + negfiles

if __name__ == '__main__':
    print mutual_info('good')
    print mutual_info('bad')
    print mutual_info('incredible')
    print mutual_info('jaskdhkasjdhkjincredible')
    feature_selection_experiment(get_paths())

