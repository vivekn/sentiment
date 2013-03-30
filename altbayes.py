"""
Train a naive Bayes classifier from the IMDb reviews data set
"""
from collections import defaultdict
from math import log
import re
import os
import random
import pickle


positive = defaultdict(int)
negative = defaultdict(int)
sums = {'pos': 0, 'neg': 0}


def tokenize(text):
    return re.findall("\w+", text)


def negate_sequence(text):
    """
    Detects negations and transforms negated words into "not_" form.
    """
    negation = False
    delims = "?.!:,;"
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


def position_info(seq):
    """ Remove duplicates and add position information """
    last_occurence = dict()
    for i, x in enumerate(seq):
        last_occurence[x] = 1
    
    return last_occurence.items()


def train():
    for path in os.listdir("./aclImdb/train/pos/"):
        path = "./aclImdb/train/pos/" + path
        words = position_info(negate_sequence(open(path).read()))
        for word, position in words:
            positive[word] += position
            sums['pos'] += position
            negative['not_' + word] += position
            sums['neg'] += position

    for path in os.listdir("./aclImdb/train/neg/"):
        path = "./aclImdb/train/neg/" + path
        words = position_info(negate_sequence(open(path).read()))
        for word, position in words:
            negative[word] += position
            sums['neg'] += position
            positive['not_' + word] += position
            sums['pos'] += position
    # data = [sums, positive, negative]
    # handle = open("trained", "wb")
    # pickle.dump(data, handle)


def get_positive_prob(word):
    return 1.0 * (positive[word] + 1) / (2 * sums['pos'])


def get_negative_prob(word):
    return 1.0 * (negative[word] + 1) / (2 * sums['neg'])



def classify(text, pneg = 0.5):
    words = position_info(negate_sequence(text))
    pscore, nscore = 0, 0

    for word, position in words:
        pscore += log(get_positive_prob(word) * position)
        nscore += log(get_negative_prob(word) * position)

    return pscore > nscore


if __name__ == '__main__':
    train()
