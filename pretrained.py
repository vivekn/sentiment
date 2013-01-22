"""
Train a naive Bayes classifier from the IMDb reviews data set
"""
from collections import defaultdict
from math import log, exp
import re
import os
import random
import pickle


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

def classify(text, pneg = 0.5):
    words = negate_sequence(text)
    pscore, nscore = 0, 0

    for word in words:
        pscore += log(get_positive_prob(word))
        nscore += log(get_negative_prob(word))

    return pscore > nscore, pscore - nscore

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

if __name__ == '__main__':
    test()